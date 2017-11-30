//! The global data and participant for garbage collection.
//!
//! # Registration
//!
//! In order to track all participants in one place, we need some form of participant
//! registration. When a participant is created, it is registered to a global lock-free
//! singly-linked list of registries; and when a participant is leaving, it is unregistered from the
//! list.
//!
//! # Pinning
//!
//! Every participant contains an integer that tells whether the participant is pinned and if so,
//! what was the global epoch at the time it was pinned. Participants also hold a pin counter that
//! aids in periodic global epoch advancement.
//!
//! When a participant is pinned, a `Guard` is returned as a witness that the participant is pinned.
//! Guards are necessary for performing atomic operations, and for freeing/dropping locations.

use core::cell::{Cell, UnsafeCell};
use core::mem;
use core::num::Wrapping;
use core::ptr;
use core::sync::atomic;
use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use alloc::boxed::Box;
use alloc::arc::Arc;

use crossbeam_utils::cache_padded::CachePadded;

use collector::Handle;
use epoch::{AtomicEpoch, Epoch};
use guard::Guard;
use garbage::{Bag, Garbage};
use sync::queue::Queue;

/// The global data for a garbage collector.
pub struct Global {
    /// The intrusive linked list of `Local`s.
    locals: AtomicPtr<LocalList>,

    /// The global queue of bags of deferred functions.
    queue: Queue<(Epoch, Bag)>,

    /// The global epoch.
    pub(crate) epoch: CachePadded<AtomicEpoch>,
}

impl Global {
    /// Number of bags to destroy.
    const COLLECT_STEPS: usize = 8;

    /// Creates a new global data for garbage collection.
    #[inline]
    pub fn new() -> Self {
        Self {
            locals: AtomicPtr::new(ptr::null_mut()),
            queue: Queue::new(),
            epoch: CachePadded::new(AtomicEpoch::new(Epoch::starting())),
        }
    }

    /// Pushes the bag into the global queue and replaces the bag with a new empty bag.
    pub fn push_bag(&self, bag: &mut Bag, guard: &Guard) {
        let bag = mem::replace(bag, Bag::new());

        atomic::fence(Ordering::SeqCst);

        let epoch = self.epoch.load(Ordering::Relaxed);
        self.queue.push((epoch, bag), guard);
    }

    /// Collects several bags from the global queue and executes deferred functions in them.
    ///
    /// Note: This may itself produce garbage and in turn allocate new bags.
    ///
    /// `pin()` rarely calls `collect()`, so we want the compiler to place that call on a cold
    /// path. In other words, we want the compiler to optimize branching for the case when
    /// `collect()` is not called.
    #[cold]
    pub fn collect(&self, guard: &Guard) {
        let global_epoch = self.try_advance();

        let condition = |item: &(Epoch, Bag)| {
            // A pinned participant can witness at most one epoch advancement. Therefore, any bag
            // that is within one epoch of the current one cannot be destroyed yet.
            global_epoch.wrapping_sub(item.0) >= 2
        };

        let steps = if cfg!(feature = "sanitize") {
            usize::max_value()
        } else {
            Self::COLLECT_STEPS
        };

        for _ in 0..steps {
            match self.queue.try_pop_if(&condition, guard) {
                None => break,
                Some(bag) => drop(bag),
            }
        }
    }

    /// Attempts to advance the global epoch.
    ///
    /// The global epoch can advance only if all currently pinned participants have been pinned in
    /// the current epoch.
    ///
    /// Returns the current global epoch.
    ///
    /// `try_advance()` is annotated `#[cold]` because it is rarely called.
    #[cold]
    pub fn try_advance(&self) -> Epoch {
        let global_epoch = self.epoch.load(Ordering::Relaxed);
        atomic::fence(Ordering::SeqCst);

        let mut ptr = self.locals.load(Ordering::Acquire);
        while let Some(list) = unsafe { ptr.as_ref() } {
            for local in list.locals.iter() {
                let local_epoch = local.epoch.load(Ordering::Relaxed);

                // If the participant was pinned in a different epoch, we cannot advance the
                // global epoch just yet.
                if local_epoch.is_pinned() && local_epoch.unpinned() != global_epoch {
                    return global_epoch;
                }
            }
            ptr = list.next.load(Ordering::Acquire);
        }
        atomic::fence(Ordering::Acquire);

        // All pinned participants were pinned in the current global epoch.
        // Now let's advance the global epoch...
        //
        // Note that if another thread already advanced it before us, this store will simply
        // overwrite the global epoch with the same value. This is true because `try_advance` was
        // called from a thread that was pinned in `global_epoch`, and the global epoch cannot be
        // advanced two steps ahead of it.
        let new_epoch = global_epoch.successor();
        self.epoch.store(new_epoch, Ordering::Release);
        new_epoch
    }

    /// Allocates a new `Local`.
    fn new_local(&self) -> *const Local {
        let mut list_head = self.locals.load(Ordering::Acquire);
        loop {
            // Look for an unused `Local` in the linked list.
            let mut ptr = list_head;
            while let Some(list) = unsafe { ptr.as_ref() } {
                for local in list.locals.iter() {
                    if !local.in_use.load(Ordering::Relaxed) &&
                       !local.in_use.swap(true, Ordering::Acquire) {
                        return local;
                    }
                }
                ptr = list.next.load(Ordering::Acquire);
            }

            // Allocate a new linked list node.
            let new = Box::new(LocalList::default());
            new.next.store(list_head, Ordering::Relaxed);
            new.locals[0].in_use.store(true, Ordering::Relaxed);
            let new = Box::into_raw(new);

            // Try to insert the new node into the linked list. If this fails
            // then it means that another thread won a race and added a new
            // node. If that is the case then we free the our node and try to
            // find a slot in the newly added node.
            match self.locals.compare_exchange(list_head, new, Ordering::Release, Ordering::Relaxed) { 
                Ok(_) => return unsafe { &(*new).locals[0] },
                Err(x) => {
                    drop(unsafe { Box::from_raw(new) });
                    list_head = x;
                }
            }
        }
    }
}

impl Drop for Global {
    fn drop(&mut self) {
        // Free the linked list of `Local`s on drop, since we can't free it
        // while `Local`s may still be added or removed.
        let mut ptr = self.locals.load(Ordering::Relaxed);
        while !ptr.is_null() {
            unsafe {
                let next = (*ptr).next.load(Ordering::Relaxed);
                drop(Box::from_raw(ptr));
                ptr = next;
            }
        }
    }
}

/// Number of `Local`s packed in each linked list entry.
const LOCALS_PER_ENTRY: usize = 32;

/// Linked list node for `Local`s in a `Global`.
#[derive(Default)]
struct LocalList {
    /// A node in the intrusive linked list of `LocalList`s.
    next: AtomicPtr<LocalList>,

    /// `Local` structs in this entry.
    locals: [Local; LOCALS_PER_ENTRY],
}

/// Participant for garbage collection.
#[cfg_attr(feature = "nightly", repr(align(128)))]
pub struct Local {
    /// A flag indicating whether this `Local` is currently in use. An unused
    /// `Local` must have an epoch value of `Epoch::starting()`.
    in_use: AtomicBool,

    /// The local epoch.
    epoch: AtomicEpoch,

    /// A reference to the global data.
    ///
    /// When all guards and handles get dropped, this reference is destroyed.
    global: Cell<*const Global>,

    /// The local bag of deferred functions.
    pub(crate) bag: UnsafeCell<Bag>,

    /// The number of guards keeping this participant pinned.
    guard_count: Cell<usize>,

    /// The number of active handles.
    handle_count: Cell<usize>,

    /// Total number of pinnings performed.
    ///
    /// This is just an auxilliary counter that sometimes kicks off collection.
    pin_count: Cell<Wrapping<usize>>,
}

unsafe impl Sync for Local {}

impl Default for Local {
    fn default() -> Self {
        Local {
            in_use: AtomicBool::new(false),
            epoch: AtomicEpoch::new(Epoch::starting()),
            global: Cell::new(ptr::null()),
            bag: UnsafeCell::new(Bag::new()),
            guard_count: Cell::new(0),
            handle_count: Cell::new(0),
            pin_count: Cell::new(Wrapping(0)),
        }
    }
}

impl Local {
    /// Number of pinnings after which a participant will execute some deferred functions from the
    /// global queue.
    const PINNINGS_BETWEEN_COLLECT: usize = 128;

    /// Registers a new `Local` in the provided `Global`.
    pub fn register(global: Arc<Global>) -> Handle {
        let local = global.new_local();
        unsafe {
            (*local).handle_count.set(1);
            (*local).global.set(Arc::into_raw(global));
        }
        Handle { local }
    }

    /// Returns a reference to the `Global` in which this `Local` resides.
    #[inline]
    pub fn global(&self) -> &Global {
        unsafe { &*self.global.get() }
    }

    /// Returns `true` if the current participant is pinned.
    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.guard_count.get() > 0
    }

    pub fn defer(&self, mut garbage: Garbage, guard: &Guard) {
        let bag = unsafe { &mut *self.bag.get() };

        while let Err(g) = bag.try_push(garbage) {
            self.global().push_bag(bag, guard);
            garbage = g;
        }
    }

    pub fn flush(&self, guard: &Guard) {
        let bag = unsafe { &mut *self.bag.get() };

        if !bag.is_empty() {
            self.global().push_bag(bag, guard);
        }

        self.global().collect(guard);
    }

    /// Pins the `Local`.
    #[inline]
    pub fn pin(&self) -> Guard {
        let guard = Guard { local: self };

        let guard_count = self.guard_count.get();
        self.guard_count.set(guard_count.checked_add(1).unwrap());

        if guard_count == 0 {
            let global_epoch = self.global().epoch.load(Ordering::Relaxed);
            let new_epoch = global_epoch.pinned();

            // Now we must store `new_epoch` into `self.epoch` and execute a `SeqCst` fence.
            // The fence makes sure that any future loads from `Atomic`s will not happen before
            // this store.
            if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
                // HACK(stjepang): On x86 architectures there are two different ways of executing
                // a `SeqCst` fence.
                //
                // 1. `atomic::fence(SeqCst)`, which compiles into a `mfence` instruction.
                // 2. `_.compare_and_swap(_, _, SeqCst)`, which compiles into a `lock cmpxchg`
                //    instruction.
                //
                // Both instructions have the effect of a full barrier, but benchmarks have shown
                // that the second one makes pinning faster in this particular case.
                let current = Epoch::starting();
                let previous = self.epoch.compare_and_swap(current, new_epoch, Ordering::SeqCst);
                debug_assert_eq!(current, previous, "participant was expected to be unpinned");
            } else {
                self.epoch.store(new_epoch, Ordering::Relaxed);
                atomic::fence(Ordering::SeqCst);
            }

            // Increment the pin counter.
            let count = self.pin_count.get();
            self.pin_count.set(count + Wrapping(1));

            // After every `PINNINGS_BETWEEN_COLLECT` try advancing the epoch and collecting
            // some garbage.
            if count.0 % Self::PINNINGS_BETWEEN_COLLECT == 0 {
                self.global().collect(&guard);
            }
        }

        guard
    }

    /// Unpins the `Local`.
    #[inline]
    pub fn unpin(&self) {
        let guard_count = self.guard_count.get();
        self.guard_count.set(guard_count - 1);

        if guard_count == 1 {
            self.epoch.store(Epoch::starting(), Ordering::Release);

            if self.handle_count.get() == 0 {
                self.finalize();
            }
        }
    }

    /// Unpins and then pins the `Local`.
    #[inline]
    pub fn repin(&self) {
        let guard_count = self.guard_count.get();

        // Update the local epoch only if there's only one guard.
        if guard_count == 1 {
            let epoch = self.epoch.load(Ordering::Relaxed);
            let global_epoch = self.global().epoch.load(Ordering::Relaxed);

            // Update the local epoch only if the global epoch is greater than the local epoch.
            if epoch != global_epoch {
                // We store the new epoch with `Release` because we need to ensure any memory
                // accesses from the previous epoch do not leak into the new one.
                self.epoch.store(global_epoch, Ordering::Release);

                // However, we don't need a following `SeqCst` fence, because it is safe for memory
                // accesses from the new epoch to be executed before updating the local epoch.  At
                // worse, other threads will see the new epoch late and delay GC slightly.
            }
        }
    }

    /// Increments the handle count.
    #[inline]
    pub fn acquire_handle(&self) {
        let handle_count = self.handle_count.get();
        debug_assert!(handle_count >= 1);
        self.handle_count.set(handle_count + 1);
    }

    /// Decrements the handle count.
    #[inline]
    pub fn release_handle(&self) {
        let guard_count = self.guard_count.get();
        let handle_count = self.handle_count.get();
        debug_assert!(handle_count >= 1);
        self.handle_count.set(handle_count - 1);

        if guard_count == 0 && handle_count == 1 {
            self.finalize();
        }
    }

    /// Removes the `Local` from the global linked list.
    #[cold]
    fn finalize(&self) {
        debug_assert_eq!(self.guard_count.get(), 0);
        debug_assert_eq!(self.handle_count.get(), 0);

        // Temporarily increment handle count. This is required so that the following call to `pin`
        // doesn't call `finalize` again.
        self.handle_count.set(1);
        unsafe {
            // Pin and move the local bag into the global queue. It's important that `push_bag`
            // doesn't defer destruction on any new garbage.
            let guard = &self.pin();
            self.global().push_bag(&mut *self.bag.get(), guard);
        }
        // Revert the handle count back to zero.
        self.handle_count.set(0);

        unsafe {
            // Take the reference to the `Global` out of this `Local`. Since we're not protected
            // by a guard at this time, it's crucial that the reference is read before marking the
            // `Local` as deleted.
            let global: Arc<Global> = Arc::from_raw(self.global.get());

            // Mark this node in the linked list as free for re-use.
            self.in_use.store(false, Ordering::Release);

            // Finally, drop the reference to the global.  Note that this might be the last
            // reference to the `Global`. If so, the global data will be destroyed and all deferred
            // functions in its queue will be executed.
            drop(global);
        }
    }
}