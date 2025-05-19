ros2 service call /check_state_validity moveit_msgs/srv/GetStateValidity "{
  robot_state: {
    joint_state: {
      name: [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
      ],
      position: [
        0.0174,
        -1.5708,
        0.0,
        -1.5708,
        0.0,
        0.0
      ]
    }
  },
  group_name: 'ur_manipulator'
}



ros2 service call /check_state_validity moveit_msgs/srv/GetStateValidity "{robot_state: {joint_state: {name: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], position: [0.0174, -1.5708, 0.0, -1.5708, 0.0, 0.0]}}, group_name: 'ur_manipulator'}"
