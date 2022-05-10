from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.linear_model import Ridge
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs


def signed_distance_to_plane(point, coeffs, bias):
	# dont use np.dot here or cpu usage will skyrocket.
	# https://www.pugetsystems.com/labs/hpc/How-To-Use-MKL-with-AMD-Ryzen-and-Threadripper-CPU-s-Effectively-for-Python-Numpy-And-Other-Applications-1637/
	a = (point[:, 0]*coeffs[0]+point[:, 1]*coeffs[1] +
		 point[:, 2]*coeffs[2]+bias)/np.linalg.norm(coeffs)
	# a = (np.matmul(point,coeffs)+bias) / np.linalg.norm(coeffs)
	return a


class RoadCurbDetector:
	"""
	this class can detect roadcurb supposedly in the range of 4m
	"""
	linear_regressor = Ridge()
	pc = rs.pointcloud()
	near_collision = False

	def __init__(self, map_resolution=0.02, map_size=5, orientation=(0, 0, 0, 1), position=(0, 0, 0),
				 sampling_radius=0.3, min_height_from_realsense=0.08, height_from_ground=0.04, emergency=False):
		self.MAP_RESOLUTION = map_resolution
		self.MAP_SIZE = map_size
		self.orientation = orientation
		self.position = position
		self.sampling_radius = sampling_radius
		self.min_height_from_realsense = min_height_from_realsense
		self.height_from_ground = height_from_ground
		self.emergency_handler_enabled = emergency

	def __pts2og(self, pts, angle_step=np.pi/180/4, field_of_view=80.0/180.0*np.pi):
		MAP_SIZE = self.MAP_SIZE
		MAP_RESOLUTION = self.MAP_RESOLUTION
		grid_size = int(MAP_SIZE/MAP_RESOLUTION)
		og = np.zeros((grid_size, grid_size))-1

		# define the minimum angle and max angle the camera can cover
		angle_min = np.pi/2 - field_of_view/2
		angle_max = np.pi/2 + field_of_view/2  # - 3/180*np.pi
		max_distance = 4  # points that are above this distance will be discarded from og
		angle_bin = np.arange(start=angle_min, stop=angle_max, step=angle_step)
		bin_num = angle_bin.shape[0]
		x_max = max_distance*np.cos(angle_bin)
		y_max = max_distance*np.sin(angle_bin)
		placeholder_pts = np.stack([x_max, y_max], axis=1)
		y = pts[:, 1]
		pts = pts[y <= max_distance]
		theta = np.arctan2(pts[:, 1], pts[:, 0])
		# print("theoery",angle_min*180/np.pi,angle_max*180/np.pi,np.min(theta)*180/np.pi,np.max(theta)*180/np.pi)
		angle_is_valid = np.logical_and(
			theta >= angle_min, theta <= angle_max)
		pts = pts[angle_is_valid]
		theta = theta[angle_is_valid]
		binned_theta = np.round(
			(theta-angle_min)/angle_step).astype(np.int32)
		dist = np.linalg.norm(pts, axis=1)
		sorted_by_theta_and_dist_index = np.lexsort((dist, binned_theta))
		# choose the points(also the index) which is the closest to the origin in that binned theta
		angle, index_new = np.unique(
			binned_theta[sorted_by_theta_and_dist_index], return_index=True)
		sorted_by_theta_and_dist_index = sorted_by_theta_and_dist_index[index_new]
		sorted_by_angle_occupied_pts = pts[sorted_by_theta_and_dist_index]
		new_pts = sorted_by_angle_occupied_pts
		#===================================================================================
		checkpoint=0
		occupied_start = np.array([0,0])
		occupied_end = np.array([0,0])
		is_occupied_region = False
		area_skipped = False
		# print("session:===================")

		# traverse the pts from left to right and remove
		for i in range(len(angle)-1):
			# print("data is" ,i,ang,checkpoint)
			# if np.linalg.norm(sorted_by_angle_occupied_pts[i]-sorted_by_angle_occupied_pts[i+1]) > threshold or \
			#     np.linalg.norm(sorted_by_angle_occupied_pts[i]-sorted_by_angle_occupied_pts[i-1]) > threshold:
			the_i_step = i+checkpoint

			# check if there is no object in this area but there is only a single point, the we discard the point and
			# replace the middle part with the placeholder pts
			if angle[i+1]-angle[i]>20:
				#print("noise point is",sorted_by_angle_occupied_pts[i],i)
				pts_before_i = new_pts[:the_i_step+1]
				pts_after_i = new_pts[the_i_step+1:] 
				inserted_placeholder_pts = placeholder_pts[angle[i]:angle[i+1]]
				new_pts = np.concatenate([pts_before_i,inserted_placeholder_pts,pts_after_i],axis=0)
				# print("shape of "+str(i)+" element: ",new_pts.shape,pts_before_i.shape,pts_after_i.shape)
				checkpoint += angle[i+1]-angle[i]
				area_skipped = True
			else:
				area_skipped = False

			# print("i and checkpoint",i,checkpoint)
			# print("length",np.linalg.norm(new_pts[the_i_step]))


			# if there is an obstacle in this angle, mark the start of this area, meaning this area is occupied 
			if np.linalg.norm(new_pts[the_i_step]) < max_distance and not is_occupied_region:
				occupied_start = new_pts[the_i_step]
				is_occupied_region = True
				# print("start angle bin is",the_i_step)
			# if there is no obstacle in the angle, or we reach the end, or there is noise in that angle and make us skip that area
			#  mark the end of that occupied area and draw an arc from the start to the end
			# because we have to make sure that there cannot be free or unknown space behind the object.
			elif (np.linalg.norm(new_pts[the_i_step])>= max_distance or i == len(angle)-2 or area_skipped) and is_occupied_region:
				occupied_end = new_pts[the_i_step-1]
				# print("end angle bin is",the_i_step-1)
				is_occupied_region = False
				angle_start = int(np.rad2deg(np.arctan2(occupied_start[1],occupied_start[0])))
				angle_end = int(np.rad2deg(np.arctan2(occupied_end[1],occupied_end[0])))
				og = cv2.ellipse(og,(int(grid_size/2),0),
							(grid_size,grid_size),
							0,angle_start,angle_end,100,-1)


		#======================================================================================
		
		# add default value to rightmost and leftmost missing angles that are not provided by laser detector.
		if len(angle) > 0:
			# print(angle[[0, -1]], bin_num)
			if angle[0] > 0:
				new_pts = np.concatenate(
					[placeholder_pts[0:angle[0]+1], new_pts])
			if angle[-1] < bin_num-1:
				# print(new_pts.shape, placeholder_pts[angle[-1]:bin_num].shape)
				new_pts = np.concatenate(
					[new_pts, placeholder_pts[angle[-1]:bin_num]], axis=0)
		else:
			new_pts = placeholder_pts

		# draw the laser points on the canvas
		new_pts = np.append(
			new_pts, np.array([[0, 0]]), axis=0)/MAP_RESOLUTION
		origin = np.array([grid_size/2, 0])
		new_pts += origin
		new_pts = new_pts.astype(np.int32)
		og = cv2.fillPoly(og, [new_pts], 0)
		# for pt in (sorted_by_angle_occupied_pts/MAP_RESOLUTION + origin).astype(np.int):
		#     cv2.circle(og, (pt[0], pt[1]), radius=1, color=100, thickness=-1)
		# og = cv2.dilate(og, np.ones((3, 3)))
		return og

	def _og_msg(self, occ_grid, map_resolution, map_size, time_stamp):
		ORIENTATION = self.orientation
		POSITION = self.position
		MAP_RESOLUTION = map_resolution  # Unit: Meter
		MAP_SIZE = map_size  # Unit: Meter, Shape: Square with center "base_link"
		map_img = cv2.rotate(
			occ_grid, cv2.ROTATE_90_COUNTERCLOCKWISE).astype(np.int8)

		occupancy_grid = map_img.flatten()
		occupancy_grid = occupancy_grid.tolist()

		map_msg = OccupancyGrid()
		map_msg.header = Header()
		map_msg.header.frame_id = "base_link"
		map_msg.header.stamp = time_stamp

		map_msg.info = MapMetaData()
		map_msg.info.height = int(MAP_SIZE / MAP_RESOLUTION)  # Unit: Pixel
		map_msg.info.width = int(MAP_SIZE / MAP_RESOLUTION)  # Unit: Pixel
		map_msg.info.resolution = MAP_RESOLUTION

		map_msg.info.origin = Pose()
		map_msg.info.origin.position = Point()
		map_msg.info.origin.position.x = POSITION[0]  # Unit: Meter
		# -MAP_SIZE / 2  # Unit: Meter
		map_msg.info.origin.position.y = POSITION[1]
		map_msg.info.origin.position.z = POSITION[2]
		map_msg.info.origin.orientation = Quaternion()
		map_msg.info.origin.orientation.x = ORIENTATION[0]
		map_msg.info.origin.orientation.y = ORIENTATION[1]
		map_msg.info.origin.orientation.z = ORIENTATION[2]
		map_msg.info.origin.orientation.w = ORIENTATION[3]
		map_msg.data.extend(occupancy_grid)
		map_msg.info.map_load_time = rospy.Time.now()
		return map_msg

	def depth_frame_2_occ_grid(self, depth_frame):

		'''
		----------
		this  function convert from depthframe to Occupancy grid
		The general steps are:
		1. Extract pointcloud data from depth frame, if camera is hanged upside down, then flip the x and y
		## in point cloud frame, the origin start from the camera and 
			-	z points forward
			-	x points right
			-	y points downward
		2. Assume the area in front of cam is empty, find the equation of the plane(ground) using Ridge
		3. Find all point in the pointcloud which has the distance to the plane = the height_from_ground param
		4. collect all the point and convert it to og messaage and return it


		'''
		MAP_SIZE = self.MAP_SIZE
		MAP_RESOLUTION = self.MAP_RESOLUTION
		points = self.pc.calculate(depth_frame)
		v, _ = points.get_vertices(), points.get_texture_coordinates()
		# xyz # vertices are in metres unit
		vtx = np.asanyarray(v).view(np.float32).reshape(-1, 3)
		# y_only is a (2, n) dimesion array, not (h,w) dimension array
		nonzero_pts_index, = np.nonzero(vtx[:, 2] != 0)
		nonzero_pts = vtx[nonzero_pts_index]

		# if you put the camera upside down then uncomment this line
		# nonzero_pts[:, [0, 1]] = -nonzero_pts[:, [0, 1]]
		z_only = nonzero_pts[:, 2]
		proximity_camera_area = (nonzero_pts[np.sqrt(
			z_only**2) < self.sampling_radius])
		proximity_pts_near_ground = proximity_camera_area[proximity_camera_area[:, 1]
														  < self.min_height_from_realsense]

		# ==========================Handle emergency control here===============================
		if self.emergency_handler_enabled:
			emergency_range = 0.4  # np.clip(np.abs(speed*0.5), 0.3, 0.5)
			emergency_area = nonzero_pts[z_only < emergency_range]
			emergency_area_y = emergency_area[emergency_area[:, 1]
											  > self.min_height_from_realsense]
			if emergency_area_y.shape[0] < emergency_area.shape[0]*9/10:
				# about to collide with another object
				self.near_collision = True
				print("EMERGENCY_STOP! ,OBJECT TOO CLOSE!")
				grid_size = int(MAP_SIZE/MAP_RESOLUTION)
				og = (np.ones((grid_size, grid_size))*100)
				msg = self._og_msg(og, MAP_RESOLUTION,
								   MAP_SIZE, rospy.Time.now())
				return msg
			self.near_collision = False
		# =======================================================================================

		train_data = proximity_pts_near_ground[:, (0, 2)]
		target = proximity_pts_near_ground[:, 1]
		try:
			linear_regressor = self.linear_regressor
			linear_regressor.fit(train_data, target)

			# finding score then retrain is much slower than training directly
			# plane_finding_acc = linear_regressor.score(train_data, target)

			coeffs = np.array([linear_regressor.coef_[0], -
							   1, linear_regressor.coef_[1]])
			bias = linear_regressor.intercept_

			# this is in point cloud coordinate frame, so z axis  points upward
			ground_approximation_mask_index, = np.nonzero(np.abs(
				signed_distance_to_plane(nonzero_pts, coeffs, bias)-self.height_from_ground) < 0.001)
			ground_approximation_plane = nonzero_pts[ground_approximation_mask_index]
			# print(ground_approximation_plane)
			pts = ground_approximation_plane[:, (0, 2)]
			og = self.__pts2og(pts)
			msg = self._og_msg(og, MAP_RESOLUTION, MAP_SIZE, rospy.Time.now())
			# pts *= 100
			# size = 500
			# a = np.zeros((size, size)).astype(np.uint8)
			# if len(ground_approximation_plane) > 0:
			# 	# print("max z ", np.max(ground_approximation_plane[:, 2]))
			# 	# print("max_x", np.max(ground_approximation_plane[:, 0]))
			# 	# print("max_y", np.max(ground_approximation_plane[:, 1]))
			# 	# print(np.histogram(pts))
			# 	pts[:, 1] *= -1
			# 	pts += np.array([size/2, size-1])
			# 	pts = np.round(pts).astype(np.int32)
			# 	for pt in pts:
			# 		a = cv2.circle(a, (pt[0], pt[1]), 1, 255, -1)
			# cv2.imshow("bev", a)

			# # this part is for illustration purpose:
			# # print(ground_approximation_mask_index, nonzero_pts_index)
			# pts_on_ground = nonzero_pts_index[ground_approximation_mask_index]
			# frame_width = 848
			# coordx_pt_on_ground = pts_on_ground % frame_width
			# coordy_pt_on_ground = pts_on_ground // frame_width
			# # print(coordx_pt_on_ground)
			# # t0 = time.time()
			# depth_frame = np.asarray(depth_frame.get_data())
			# # print(depth_frame.shape)
			# # depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
			# depth_frame = cv2.cvtColor(
			# 	(depth_frame / 8000 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
			# for i in range(len(pts_on_ground)):
			# 	cv2.circle(
			# 		depth_frame, (coordx_pt_on_ground[i], coordy_pt_on_ground[i]), 1, (255, 0, 0), -1)
			# cv2.imshow("depth frame", depth_frame.astype(np.uint8))
			return msg
		except (ValueError) as e:
			print(e)
		
