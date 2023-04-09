use crate::postprocess::Category;
use std::fmt::{Display, Formatter};

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

/// The hand landmarks detection result from HandLandmark, where each vector
/// element represents a single hand detected in the image.
pub struct HandLandmarkResult {
    /// Classification of handedness.
    pub handedness: Vec<Category>,
    /// Detected hand landmarks in normalized image coordinates.
    pub hand_landmarks: Vec<super::landmark::NormalizedLandmarks>,
    /// Detected hand landmarks in world coordinates.
    pub hand_world_landmarks: Vec<super::landmark::Landmarks>,
}

impl HandLandmark {
    #[inline(always)]
    pub fn name(&self) -> &'static str {
        match self {
            HandLandmark::WRIST => "WRIST",
            HandLandmark::ThumbCmc => "THUMB_CMC",
            HandLandmark::ThumbMcp => "THUMB_MCP",
            HandLandmark::ThumbIp => "THUMB_IP",
            HandLandmark::ThumbTip => "THUMB_TIP",
            HandLandmark::IndexFingerMcp => "INDEX_FINGER_MCP",
            HandLandmark::IndexFingerPip => "INDEX_FINGER_PIP",
            HandLandmark::IndexFingerDip => "INDEX_FINGER_DIP",
            HandLandmark::IndexFingerTip => "INDEX_FINGER_TIP",
            HandLandmark::MiddleFingerMcp => "MIDDLE_FINGER_MCP",
            HandLandmark::MiddleFingerPip => "MIDDLE_FINGER_PIP",
            HandLandmark::MiddleFingerDip => "MIDDLE_FINGER_DIP",
            HandLandmark::MiddleFingerTip => "MIDDLE_FINGER_TIP",
            HandLandmark::RingFingerMcp => "RING_FINGER_MCP",
            HandLandmark::RingFingerPip => "RING_FINGER_PIP",
            HandLandmark::RingFingerDip => "RING_FINGER_DIP",
            HandLandmark::RingFingerTip => "RING_FINGER_TIP",
            HandLandmark::PinkyMcp => "PINKY_MCP",
            HandLandmark::PinkyPip => "PINKY_PIP",
            HandLandmark::PinkyDip => "PINKY_DIP",
            HandLandmark::PinkyTip => "PINKY_TIP",
        }
    }
}

impl Display for HandLandmark {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Display for HandLandmarkResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HandLandmarkResult:")?;
        if self.handedness.is_empty() {
            writeln!(f, "  No Handedness")?;
        } else {
            writeln!(f, "Handedness: ")?;
            if self.handedness.is_empty() {
                writeln!(f, "    No Categories")?;
            } else {
                for (id, c) in self.handedness.iter().enumerate() {
                    writeln!(f, "    Category #{}:", id)?;
                    write!(f, "{}", c)?;
                }
            }
        }

        writeln!(f, "  Landmarks:")?;
        if self.hand_landmarks.is_empty() {
            writeln!(f, "  No Landmarks")?;
        } else {
            for (id, l) in self.hand_landmarks.iter().enumerate() {
                writeln!(f, "    Landmark #{}:", id)?;
                write!(f, "{}", l)?;
            }
        }

        writeln!(f, "  WorldLandmarks:")?;
        if self.hand_world_landmarks.is_empty() {
            writeln!(f, "  No WorldLandmarks")?;
        } else {
            for (id, l) in self.hand_world_landmarks.iter().enumerate() {
                writeln!(f, "    Landmark #{}:", id)?;
                write!(f, "{}", l)?;
            }
        }
        Ok(())
    }
}
