/// Used for stream data results, such video, audio.
pub struct ResultsIter<'session, 'tensor, TaskSession, VideoData>
where
    TaskSession: crate::tasks::vision::TaskSession + 'session,
    VideoData: crate::preprocess::vision::VideoData,
{
    video_data: VideoData,
    session: &'session mut TaskSession,
    _marker: std::marker::PhantomData<&'tensor ()>,
}

impl<'session, 'tensor, TaskSession, VideoData>
    ResultsIter<'session, 'tensor, TaskSession, VideoData>
where
    TaskSession: crate::tasks::vision::TaskSession + 'session,
    VideoData: crate::preprocess::vision::VideoData,
{
    #[inline(always)]
    pub(crate) fn new(session: &'session mut TaskSession, video_data: VideoData) -> Self {
        Self {
            video_data,
            session,
            _marker: Default::default(),
        }
    }

    /// poll next result
    #[inline(always)]
    pub fn next(&mut self) -> Result<Option<TaskSession::Result>, crate::Error> {
        self.session.process_next(&mut self.video_data)
    }

    results_iter_impl!();
}
