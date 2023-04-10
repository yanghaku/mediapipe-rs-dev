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
pub struct ResultsIter<'session, 'tensor, TaskSession, ToTensorIterator>
where
    TaskSession: crate::tasks::TaskSession + 'session,
    ToTensorIterator: crate::preprocess::TensorsIterator + 'tensor,
{
    input_tensors_iter: ToTensorIterator,
    session: &'session mut TaskSession,
    _marker: std::marker::PhantomData<&'tensor ()>,
}

impl<'session, 'tensor, TaskSession, ToTensorIterator>
    ResultsIter<'session, 'tensor, TaskSession, ToTensorIterator>
where
    TaskSession: crate::tasks::TaskSession + 'session,
    ToTensorIterator: crate::preprocess::TensorsIterator + 'tensor,
{
    #[inline(always)]
    pub(crate) fn new(
        session: &'session mut TaskSession,
        input_tensors_iter: ToTensorIterator,
    ) -> Self {
        Self {
            input_tensors_iter,
            session,
            _marker: Default::default(),
        }
    }

    /// poll next result
    #[inline(always)]
    pub fn next(&mut self) -> Result<Option<TaskSession::Result>, crate::Error> {
        self.session.process_next(&mut self.input_tensors_iter)
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
    pub fn to_vec(mut self) -> Result<Vec<TaskSession::Result>, crate::Error> {
        let mut ans = Vec::new();
        while let Some(r) = self.next()? {
            ans.push(r);
        }
        Ok(ans)
    }
}
