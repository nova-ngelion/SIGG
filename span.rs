#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
}

impl Span {
    pub fn new(start: usize, end: usize, line: usize) -> Self {
        Self { start, end, line }
    }
    pub fn combine(a: Span, b: Span) -> Self {
        Self { start: a.start, end: b.end, line: a.line }
    }
}
