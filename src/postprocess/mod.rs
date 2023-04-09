/// result containers
mod containers;
pub use containers::*;

/// stateless operators for tensor
mod ops;
pub use ops::QuantizationParameters;

/// stateful objects, convert tensor to results
mod processing;
pub(crate) use processing::*;

/// utils to use the results, such as draw_utils
pub mod utils;

/// Used for stream data results, such video, audio.
pub struct ResultsIter<TaskSession, ToTensorIterator>
where
    TaskSession: crate::tasks::TaskSession,
    ToTensorIterator: crate::preprocess::TensorsIterator,
{
    input_tensors_iter: ToTensorIterator,
    _marker: std::marker::PhantomData<TaskSession>,
}

impl<TaskSession, ToTensorIterator> ResultsIter<TaskSession, ToTensorIterator>
where
    TaskSession: crate::tasks::TaskSession,
    ToTensorIterator: crate::preprocess::TensorsIterator,
{
    #[inline(always)]
    pub(crate) fn new(input_tensors_iter: ToTensorIterator) -> Self {
        Self {
            input_tensors_iter,
            _marker: Default::default(),
        }
    }

    /// poll next result
    #[inline(always)]
    pub fn next(
        &mut self,
        session: &mut TaskSession,
    ) -> Result<Option<TaskSession::Result>, crate::Error> {
        session.process_next(&mut self.input_tensors_iter)
    }

    /// poll all results
    #[inline(always)]
    pub fn collect<B: FromIterator<TaskSession::Result>>(self) -> B
    where
        Self: Sized,
    {
        todo!()
    }

    /// poll all results and save to [`Vec`]
    #[inline(always)]
    pub fn to_vec(
        mut self,
        session: &mut TaskSession,
    ) -> Result<Vec<TaskSession::Result>, crate::Error> {
        let mut ans = Vec::new();
        while let Some(r) = self.next(session)? {
            ans.push(r);
        }
        Ok(ans)
    }
}
