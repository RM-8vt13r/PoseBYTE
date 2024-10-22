import numpy as np
import scipy.linalg

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking poses in image space.

    The 8K-dimensional state space (K = number of keypoints. x,y,vx,vy,ax,ay,jx,jy per keypoint -> dimension 8K)

        x1, y1, x2, y2, ..., vx1, vy1, vx2, vy2, ..., ax1, ay1, ax2, ay2, ..., jx1, jy1, jx2, jy2, ...

    contains the keypoint positions (x, y) and their respective velocities, accelerations, and jerks.

    Object motion follows a velocity-acceleration-jerk model. The pose location
    (x1, y1, x2, y2, ...) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self, keypoint_confidence_threshold, acceleration_memory_factor=0, jerk_memory_factor=0):
        # Number of keypoints is not known -> don't initialize Kalman filter model matrices yet.
        self._transition_mat = None
        self._observation_mat = None
        self._keypoint_confidence_threshold = keypoint_confidence_threshold
        self._acceleration_memory_factor = acceleration_memory_factor
        self._jerk_memory_factor = jerk_memory_factor

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self._std_weight_acceleration = 1. / 160
        self._std_weight_jerk = 1. / 160

    def initiate(self, measured_position, measured_confidence):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measured_position : ndarray
            Keypoint coordinates (x1, y1, x2, y2, ...).
        measured_confidence : ndarray
            Keypoint confidences

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (4K dimensional) and covariance matrix (4Kx4K
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        # If this is the first run, initialize Kalman filter model matrices.
        self._K = int(measured_position.shape[0]/2)
        
        self._transition_mat = np.array([ # Transition from [p, v, a+1, j+1] to [p+1, v+1, a+1, j+1]
            [1, 1, 1/2, 1/6],
            [0, 1,   1, 1/2],
            [0, 0,   1,   0],
            [0, 0,   0,   1]
        ])
        self._transition_mat = self._transition_mat @ np.array([ # Transition from [p, v, a, j] to [p, v, a+1, j+1] (total: [p, v, a, j] to [p+1, v+1, a+1, j+1])
            [1, 0,                                0,                        0],
            [0, 1,                                0,                        0],
            [0, 0, self._acceleration_memory_factor, self._jerk_memory_factor],
            [0, 0,                                0, self._jerk_memory_factor]
        ])
        
        self._observation_mat = np.eye(2*self._K, 8*self._K) # Velocity, acceleration and jerk are not measured
        
        # Reduce matrix size for fps gain if possible
        if self._acceleration_memory_factor==0 and self._jerk_memory_factor==0:
            self._transition_mat = self._transition_mat[:2,:2]
            self._observation_mat = self._observation_mat[:, :4*self._K]
        elif self._jerk_memory_factor == 0:
            self._transition_mat = self._transition_mat[:3,:3]
            self._observation_mat = self._observation_mat[:, :6*self._K]
        
        self._transition_mat = np.kron(self._transition_mat, np.eye(2*self._K)) # Kronecker product to go from pose to keypoint space
        
        # Create track
        state_mean_position = measured_position
        state_mean_velocity = np.zeros_like(state_mean_position)
        state_mean_acceleration = np.zeros_like(state_mean_position) if self._acceleration_memory_factor > 0 or self._jerk_memory_factor > 0 else np.array([])
        state_mean_jerk = np.zeros_like(state_mean_position) if self._jerk_memory_factor > 0 else np.array([])
        #state_mean = np.r_[state_mean_position, state_mean_velocity]
        state_mean = np.r_[state_mean_position, state_mean_velocity, state_mean_acceleration, state_mean_jerk]
        
        if sum((measured_confidence >= self._keypoint_confidence_threshold) & (measured_confidence > 0)) > 0:
            height = max(measured_position[1::2][(measured_confidence >= self._keypoint_confidence_threshold) & (measured_confidence > 0)])-\
                     min(measured_position[1::2][(measured_confidence >= self._keypoint_confidence_threshold) & (measured_confidence > 0)]) # Scale uncertainty with pose scale
        else:
            height = 0
        
        #std = np.zeros(shape=(4*self._K,), dtype=float)
        std = np.zeros_like(state_mean, dtype=float)
        unreliable_mask = (measured_confidence.repeat(2) < self._keypoint_confidence_threshold) | (measured_confidence.repeat(2) <= 0)
        std[:2*self._K] = 2*self._std_weight_position * height # High initial uncertainty
        std[:2*self._K][unreliable_mask] = 10000 # Completely unreliable measurement -> don't impose any prior from it
        std[2*self._K:4*self._K] = 10*self._std_weight_velocity * height
        std[2*self._K:4*self._K][unreliable_mask] = 10000
        if self._acceleration_memory_factor > 0 or self._jerk_memory_factor > 0:
            std[4*self._K:6*self._K] = 15*self._std_weight_acceleration * height
            std[4*self._K:6*self._K][unreliable_mask] = 10000
        if self._jerk_memory_factor > 0:
            std[6*self._K:] = 20*self._std_weight_jerk * height
            std[6*self._K:][unreliable_mask] = 10000
        
        state_covariance = np.diag(np.square(std))
        return state_mean, state_covariance

    def predict(self, state_mean, state_covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        state_mean : ndarray
            The 8K dimensional mean vector of the object state at the previous
            time step.
        state_covariance : ndarray
            The 8Kx8K dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities, accelerations or jerks are initialized to 0 mean.

        """
        height = max(state_mean[:int(len(state_mean)/2)][1::2])-min(state_mean[:int(len(state_mean)/2)][1::2]) # Scale uncertainty with pose scale
        
        std_position = np.array([self._std_weight_position * height]*2*self._K) # Scale uncertainty with pose height. Standard deviation is added on each passed frame.
        std_velocity = np.array([self._std_weight_velocity * height]*2*self._K)
        std_acceleration = np.array([self._std_weight_acceleration * height]*2*self._K) if self._acceleration_memory_factor > 0 or self._jerk_memory_factor > 0 else np.array([])
        std_jerk = np.array([self._std_weight_jerk * height]*2*self._K) if self._jerk_memory_factor > 0 else np.array([])
        #process_noise_covariance = np.diag(np.square(np.r_[std_position, std_velocity]))
        process_noise_covariance = np.diag(np.square(np.r_[std_position, std_velocity, std_acceleration, std_jerk]))

        #state_mean = (self._transition_mat@state_mean[:,None])[:,0]
        #state_covariance = self._transition_mat@state_covariance@self._transition_mat.T + process_noise_covariance
        state_mean = (self._transition_mat@state_mean[:,None])[:,0]
        state_covariance = self._transition_mat@state_covariance@self._transition_mat.T + process_noise_covariance
        
        return state_mean, state_covariance
    
    def multi_predict(self, state_mean, state_covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        state_mean : ndarray
            The Nx8K dimensional mean matrix of the object states at the previous
            time step.
        state_covariance : ndarray
            The Nx8Kx8K dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        height = np.max(state_mean[:,:int(state_mean.shape[1]/2)][:,1::2], axis=1)-np.min(state_mean[:,:int(state_mean.shape[1]/2)][:,1::2], axis=1) # Scale uncertainty with pose scale
        std_position = np.tile(self._std_weight_position * height[:,None], (1,2*self._K)) # Scale uncertainty with pose height. Standard deviation is added on each passed frame.
        std_velocity = np.tile(self._std_weight_velocity * height[:,None], (1,2*self._K))
        std_acceleration = np.tile(self._std_weight_acceleration * height[:,None], (1,2*self._K)) if self._acceleration_memory_factor > 0 or self._jerk_memory_factor > 0 else np.zeros(shape=(height.shape[0],0), dtype=float)
        std_jerk = np.tile(self._std_weight_jerk * height[:,None], (1,2*self._K)) if self._jerk_memory_factor > 0 else np.zeros(shape=(height.shape[0], 0), dtype=float)
        #process_noise_covariance = np.array([np.diag(np.square(np.r_[std_position[n], std_velocity[n]])) for n in range(len(std_position))])
        process_noise_covariance = np.array([np.diag(np.square(np.r_[std_position[n], std_velocity[n], std_acceleration[n], std_jerk[n]])) for n in range(len(std_position))])

        state_mean = (self._transition_mat@state_mean[:,:,None])[:,:,0]
        state_covariance = self._transition_mat@state_covariance@self._transition_mat.T + process_noise_covariance
        
        return state_mean, state_covariance

    def update(self, state_mean, state_covariance, measured_position, measured_confidence):
        """Run Kalman filter correction step.

        Parameters
        ----------
        state_mean : ndarray
            The predicted state's mean vector (8K dimensional).
        covariance : ndarray
            The state's covariance matrix (8Kx8K dimensional).
        measured_position : ndarray
            The 2K dimensional measurement vector (x1, y1, ...)
        measured_confidence : ndarray
            The K dimensional confidence vector per keypoint

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # Start by projecting the state onto measurement space
        measured_keypoint_mask = (measured_confidence.repeat(2) >= self._keypoint_confidence_threshold) & (measured_confidence.repeat(2) > 0)
        if not np.sum(measured_keypoint_mask): return state_mean, state_covariance # No valid measurement -> don't update
        
        valid_observation_mat = self._observation_mat[measured_keypoint_mask]
        valid_measured_position = measured_position[measured_keypoint_mask]
        
        height = max(valid_measured_position[1::2])-min(valid_measured_position[1::2]) # Scale uncertainty with pose scale
        std_measured_position = np.array([self._std_weight_position * height]*valid_observation_mat.shape[0]) # Scale uncertainty with pose height
        observation_noise_covariance = np.diag(np.square(std_measured_position))
        
        projected_state_mean = (valid_observation_mat@state_mean[:,None])[:,0]
        projected_state_covariance = valid_observation_mat@state_covariance@valid_observation_mat.T + observation_noise_covariance
        
        # Calculate innovation, kalman gain, and new state
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_state_covariance, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(state_covariance, valid_observation_mat.T).T,
            check_finite=False).T
        innovation = valid_measured_position - projected_state_mean

        new_state_mean = state_mean + innovation@kalman_gain.T
        new_state_covariance = state_covariance - kalman_gain@projected_state_covariance@kalman_gain.T
        return new_state_mean, new_state_covariance
