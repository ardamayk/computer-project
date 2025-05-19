1. 
/check_state_validity servisi icerisinde ekstra olarak cost_source var. 

Bu cost_source'lar robotun carpisma yapmadigini ancak yakininda nesneler oldugunu belirtiyor.


# Robotun yakininda cisim var.
(.venv) r@amdu2204:~/computer-project$ ros2 service call /check_state_validity moveit_msgs/srv/GetStateValidity "{robot_state: {joint_state: {name: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], position: [0.0, 0.01745329, 0.0, 0.0, 0.0, 0.0]}}, group_name: 'ur_manipulator'}"
waiting for service to become available...
requester: making request: moveit_msgs.srv.GetStateValidity_Request(robot_state=moveit_msgs.msg.RobotState(joint_state=sensor_msgs.msg.JointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id=''), name=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], position=[0.0, 0.01745329, 0.0, 0.0, 0.0, 0.0], velocity=[], effort=[]), multi_dof_joint_state=sensor_msgs.msg.MultiDOFJointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id=''), joint_names=[], transforms=[], twist=[], wrench=[]), attached_collision_objects=[], is_diff=False), group_name='ur_manipulator', constraints=moveit_msgs.msg.Constraints(name='', joint_constraints=[], position_constraints=[], orientation_constraints=[], visibility_constraints=[]))

response:
moveit_msgs.srv.GetStateValidity_Response(valid=True, contacts=[], cost_sources=[moveit_msgs.msg.CostSource(cost_density=1.0, aabb_min=geometry_msgs.msg.Vector3(x=0.7625597546397163, y=0.0695938452724559, z=-0.00490299478729949), aabb_max=geometry_msgs.msg.Vector3(x=0.8640533616444441, y=0.19362832802842103, z=-0.00010000000000000026))], constraint_result=[])


# robotun yakininda cisim yok.
(.venv) r@amdu2204:~/computer-project$ ros2 service call /check_state_validity moveit_msgs/srv/GetStateValidity "{robot_state: {joint_state: {name: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}, group_name: 'ur_manipulator'}"
waiting for service to become available...
requester: making request: moveit_msgs.srv.GetStateValidity_Request(robot_state=moveit_msgs.msg.RobotState(joint_state=sensor_msgs.msg.JointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id=''), name=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], position=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], velocity=[], effort=[]), multi_dof_joint_state=sensor_msgs.msg.MultiDOFJointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id=''), joint_names=[], transforms=[], twist=[], wrench=[]), attached_collision_objects=[], is_diff=False), group_name='ur_manipulator', constraints=moveit_msgs.msg.Constraints(name='', joint_constraints=[], position_constraints=[], orientation_constraints=[], visibility_constraints=[]))

response:
moveit_msgs.srv.GetStateValidity_Response(valid=True, contacts=[], cost_sources=[], constraint_result=[])

(.venv) r@amdu2204:~/computer-project$ 