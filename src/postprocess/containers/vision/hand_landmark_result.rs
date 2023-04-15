use crate::postprocess::Category;
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};

/// The 21 hand landmarks.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
#[repr(C)]
pub enum HandLandmark {
    WRIST = 0,
    ThumbCmc = 1,
    ThumbMcp = 2,
    ThumbIp = 3,
    ThumbTip = 4,
    IndexFingerMcp = 5,
    IndexFingerPip = 6,
    IndexFingerDip = 7,
    IndexFingerTip = 8,
    MiddleFingerMcp = 9,
    MiddleFingerPip = 10,
    MiddleFingerDip = 11,
    MiddleFingerTip = 12,
    RingFingerMcp = 13,
    RingFingerPip = 14,
    RingFingerDip = 15,
    RingFingerTip = 16,
    PinkyMcp = 17,
    PinkyPip = 18,
    PinkyDip = 19,
    PinkyTip = 20,
}

impl HandLandmark {
    pub const NAMES: &'static [&'static str] = &[
        "WRIST",
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    ];

    // reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    pub const CONNECTIONS: &'static [(usize, usize)] = &[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ];

    #[inline(always)]
    pub fn name(&self) -> &'static str {
        Self::NAMES[(*self) as u32 as usize]
    }
}

impl Display for HandLandmark {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A single hand landmark detection result.
#[derive(Debug)]
pub struct HandLandmarkResult {
    /// Classification of handedness.
    pub handedness: Category,
    /// Detected hand landmarks in normalized image coordinates.
    pub hand_landmarks: super::landmark::NormalizedLandmarks,
    /// Detected hand landmarks in world coordinates.
    pub hand_world_landmarks: super::landmark::Landmarks,
}

/// The hand landmarks detection result from HandLandmark
#[derive(Debug)]
pub struct HandLandmarkResults(pub Vec<HandLandmarkResult>);

impl Deref for HandLandmarkResults {
    type Target = Vec<HandLandmarkResult>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HandLandmarkResults {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for HandLandmarkResults {
    type Item = HandLandmarkResult;
    type IntoIter = std::vec::IntoIter<HandLandmarkResult>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Display for HandLandmarkResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Handedness: ")?;
        writeln!(f, "    Category #0:")?;
        write!(f, "{}", self.handedness)?;
        writeln!(f, "  Landmarks:")?;
        for (i, l) in self.hand_landmarks.iter().enumerate() {
            writeln!(
                f,
                "    Normalized Landmark #{} ({}):",
                i,
                HandLandmark::NAMES[i]
            )?;
            write!(f, "{}", l)?;
        }
        writeln!(f, "  WorldLandmarks:")?;
        for (i, l) in self.hand_world_landmarks.iter().enumerate() {
            writeln!(f, "    Landmark #{} ({}):", i, HandLandmark::NAMES[i])?;
            write!(f, "{}", l)?;
        }
        Ok(())
    }
}

impl Display for HandLandmarkResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            writeln!(f, "No HandLandmarkResult.")?;
        } else {
            for (i, r) in self.iter().enumerate() {
                writeln!(f, "HandLandmarkResult #{}", i)?;
                write!(f, "{}", r)?;
            }
        }
        Ok(())
    }
}
