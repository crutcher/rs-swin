/// Stage that randomly selects one of its children.
mod choose_one;
/// Stage that does nothing.
pub mod noop;
/// Stage that runs a sequence of stages.
pub mod sequence;
/// Stage that randomly runs, or skips, a child.
mod with_prob;
