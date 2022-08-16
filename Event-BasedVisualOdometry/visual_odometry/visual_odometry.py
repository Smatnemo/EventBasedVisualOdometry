import matplotlib.pyplot as plt
import numpy as np
#import from modules


def motion_estimater(ms, keypoints1, keypoints2, k, stereodepth1, maximum_depth):
    
    rotationmat = np.eye(3)
    translationvector = np.zeros((3,1))
    
    img1_points = np.float32([keypoints1[m.queryIdx].pt for m in ms])
    img2_points = np.float32([keypoints2[m.trainIdx].pt for m in ms])
    
    c_x = k[0,2]
    c_y = k[1,2]
    f_x = k[0,0]
    f_y = k[1,1]
    
    points_of_objects = np.zeros((0, 3))
    deleted_points = []
    
    for i, (u,v) in enumerate(img1_points):
        z = stereodepth1[int(round(v)), int(round(u))]
        if z > maximum_depth:
            deleted_points.append(i)
            continue
            
        x = z*(u-c_x)/f_x
        y = z*(v-c_y)/f_y
        
        points_of_objects = np.vstack([points_of_objects, np.array([x, y, z])])
        # points_of_objects = np.vstack([points_of_objects, np.linalg.inv(k).dot(z*np.array([u,v,1]))])
    
    img1_points = np.delete(img1_points, deleted_points, 0)
    img2_points = np.delete(img2_points, deleted_points, 0)
    
    _, rvec, tvec, inliers = cv2.solvePnPRansac(points_of_objects, img2_points, k, None)
    
    rmat = cv2.Rodrigues(rvec)[0]
    
    return rmat, tvec, img1_points, img2_points


def visual_odometry(data, det='sift', matchmethod='BF', filt_match_distance=None,
                   stereomatcher='sgbm', mask=None, subset=None, plot=False):
    # Determine if the data has lidar data
    lidar = data.lidar
    
    #Report methods being used to user
    print('Generating disparities with Stereo()'.format(str.upper(stereomatcher)))
    print('Detecting features with{} and matching with {}'.format(str.upper(det), matchmethod))
    
    if filt_match_distance is not None:
        print('filtering feature matches at threshold of {}*distance'.format(filt_match_distance))
    if lidar:
        print('Improving stereo depth estimatation with lidar data')
    if subset is not None:
        num_frames=subset
    else:
        num_frames=data.num_frames
        
    if plot:
        fig = plt.figure(figsize=(14,14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = data.gt[:, 0, 3]
        ys = data.gt[:, 1, 3]
        zs = data.gt[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='k')
        
    #Establish a homogeneous transformation matrix. First pose is identity
    T_total = np.eye(4)
    pose_estimates = np.zeros((num_frames, 3, 4))
    pose_estimates[0] =T_total[:3, :]
    imheight =data.imheight
    imwidth =data.imwidth
    
    # Decompose left camera projection matrix to get intrinsic k matrix
    k0, r0, t0 = decompose_projection_matrix(data.P0)
    
    if data.low_memory:
        data.reset_frames()
        next_left_image = next(data.left_images)
        
    #Iterate through all frames of the sequence
    for i in range(num_frames-1):
        start = datetime.datetime.now()
        
        
        if data.low_memory:
            left_image = next_left_image
            next_left_image = next(data.left_images)
            right_image = next(data.right_images)
        else:
            left_image = data.left_images[i]
            next_left_image = data.left_images[i+1]
            right_image = data.right_images[i]
            
        depth = stereo_2_depth(left_image,
                              right_image,
                              P0= data.P0,
                              P1 = data.P1,
                              matcher=stereomatcher
                              )
        
        if lidar:
            if data.low_memory:
                pointcloud = next(data.pointclouds)
            else:
                pointcloud = data.pointclouds[i]
                
            lidar_depth = pointcloud2image(pointcloud,
                                          imheight= imheight,
                                          imwidth=imwidth,
                                          Tr=data.Tr,
                                          P0=data.P0
                                          )
            #indices = np.where(lidar_depth > 0)
            #depth[indices] = lidar_depth[indices]
            
            
        # Get features and descriptors for two sequential left frames
        features0, descriptors0 = feature_extractor(left_image, det, mask)
        features1, descriptors1 = feature_extractor(next_left_image, det, mask)
        
        unfiltered_matches = feature_matcher(descriptors0,
                                            descriptors1,
                                            matchmethod =matchmethod,
                                            det=det
                                            )
        #Filter matches based on ratio test
        if filt_match_distance is not None:
            matches = matches_filter(unfiltered_matches, filt_match_distance)
        else:
            matches = unfiltered_matches
            
        #Estimate motion between sequential images of the left camera
        rmat, tvec, img1pts, img2pts = motion_estimater(matches,
                                                       features0,
                                                       features1,
                                                       k0,
                                                       depth,
                                                       1000)
        #print('Matches before filtering:', len(unfiltered_matches))
        #print('Matches after filtering:', len(matches))
        
        #Create a blank homogeneous transformation matrix
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_total = T_total.dot(np.linalg.inv(Tmat))
        
        
        pose_estimates[i+1, :, :] = T_total[:3, :]
        
        end = datetime.datetime.now()
        print('Time to compute frame {}:'.format(i+1), end-start)
        
        
        if plot:
            xs = pose_estimates[:i+2, 0, 3]
            ys = pose_estimates[:i+2, 1, 3]
            zs = pose_estimates[:i+2, 2, 3]
            plt.plot(xs, ys, zs, c= 'red')
            plt.pause(1e-32)
        
    if plot:
        plt.close()
            
            
    return pose_estimates
            