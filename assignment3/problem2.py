import numpy as np
import math
import random

class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """

        
        # Create an numpy array in shape(n,m)
        distance = np.zeros((features2.shape[1],features1.shape[1]))

        for i in range(0,features2.shape[1]):
            for j in range(0, features1.shape[1]):
                # distance = (v1 - v2)^2
                distance[i][j] = np.sum(np.square(features2[:,i] - features1[:,j]))

        return distance
    #
    # You code here
    #

    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """

        m = p1.shape[0]
        n = p2.shape[0]
        min_1 = min(m,n)
        pairs = np.zeros((min_1,4))
        """
        Sort the distance  of p1 and p2. The function argsort will return the index of data.
        
        e.g 
        [[8,4,6]   sort with axis = 0   [[1,0,1]
         [1,9,5]            =>           [2,2,0]
         [2,7,8]]                        [0,1,2]]
        
        Since m < n, we choose the first row of the argsort array, which is the minimum distance
        """
        min_match = distances.argsort(axis = 0)[0]
        
        # Put the points with minumum distance into array pair
        for i in range(min_1):
            index = min_match[i]
            pairs[i][0] = p1[i][0]
            pairs[i][1] = p1[i][1]
            pairs[i][2] = p2[index][0]
            pairs[i][3] = p2[index][1]

        return pairs
    #
    # You code here
    #


    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """
        # Use random.sample to pick the points
        index = random.sample(range(p1.shape[0]), k)

        s1 = []
        s2 = []
        for i in index:
            s1.append([p1[i][0],p1[i][1]])
            s2.append([p2[i][0],p2[i][1]])

        s1 = np.array(s1)
        s2 = np.array(s2)


        return s1,s2
    #
    # You code here
    #


    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """
        # Add 1 at the end of each row. Each row "points array" will be transform from [x,y] to [x,y,1] 
        ps = np.zeros((points.shape[0],3))
        for i in range(0, points.shape[0]):
            ps[i][0] = points[i][0]
            ps[i][1] = points[i][1]
            ps[i][2] = 1

        # Use argsort to sort the array and get the index.
        sort = ps.argsort(axis = 0)
        
        # Shift and scale.
        s = 0.5 * ps[sort[-1][0]][0]
        tx = ps[:,0].sum()/ps.shape[0]
        ty = ps[:,1].sum()/ps.shape[0]
       
        
       
        T = [[1/s,0,-(tx/s)],[0,1/s,-(ty/s)],[0,0,1]]
        T= np.array(T)
        
        new_ps = []
        # Calculate the new points
        for i in range(0,points.shape[0]):
            new_ps.append(T@ps[i])
        new_ps = np.array(new_ps)
        
        return new_ps,T
    #
    # You code here
    #


    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices shoul be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """
        A1 = []
        A2 = []
        # Create the A array
        for i in range(0,p1.shape[0]):
            r1 = [0, 0, 0, p1[i][0], p1[i][1], p1[i][2], -(p1[i][0]*p2[i][1]), -(p1[i][1]*p2[i][1]), -(p2[i][1])]
            r2 = [-p1[i][0], -p1[i][1], -p1[i][2], 0, 0, 0, p1[i][0]*p2[i][0], p1[i][1]*p2[i][0], p2[i][0]]
            A1.append(r1)
            A1.append(r2)
        A1 = np.array(A1)

        
        # Use SVD to get the vector
        U, s, V = np.linalg.svd(A1)
        HC = V[-1]
        HC = HC.reshape(3,3)

        
        # Set the bottom right element into 1
        HCn = HC / HC[2][2]

        
        # Calculate original H
        H = np.linalg.inv(T2) @ HC @ T1

        
        # Set the bottom right element into 1
        Hn = H / H[2][2]

        return Hn,HCn
    #
    # You code here
    #


    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """
        # Add 1 at the end of each row. Each row "points array" will be transform from [x,y] to [x,y,1] 
        all_p = np.concatenate( (p,np.ones((p.shape[0], 1))), axis = 1)
        est_p = np.zeros((p.shape[0],2))
        # Compute the transform of the points.
        for i in range(0,p.shape[0]):
            temp = np.dot(H,all_p[i])
            if temp[2] == 0:
                est_p[i] = (temp/0.0001)[0:2]
            else:
                # Modify the result from [x,y,z] to [x/z,y/z,1]
                est_p[i] = (temp/temp[2])[0:2]
            
        return est_p
    #
    # You code here
    #


    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
        
        
        # Calculate the transformation of p1 and p2.
        Hx1 = self.transform_pts(p1, H)
        Hx2 = self.transform_pts(p2, np.linalg.inv(H))
        d = []
        #Calculate the distance
        for i in range(0,Hx1.shape[0]):
            diff1 = np.linalg.norm(Hx1[i] - p2[i])
            diff2 = np.linalg.norm(p1[i] - Hx2[i])

            d.append(diff1 + diff2)

        d = np.array(d)

        return d
    #
    # You code here
    #


    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
        # Get the index with distance under the threshold
        idx = np.where(dist < threshold)[0]
        
        #Number of inliers
        N = len(idx)
        #print(N)
        inliers = pairs[idx]
        inliers = np.array(inliers)

        return N,inliers
    #
    # You code here
    #


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
        """
        Formula in the slide
        """
        iters = int(math.log(1 - z) / math.log(1 - p**k))
        return iters
    #
    # You code here
    #



    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """

        p1 = pairs[:,0:2]
        p2 = pairs[:,2:4]
        
        best_N = 0
        best_inliers = []
        best_H = []
        
        
        
        for i in range(0,n_iters):
            #Pick samples
            s1,s2 = self.pick_samples(p1,p2,k)
            # Condtioning the samples
            ps1,T1 = self.condition_points(s1)
            ps2,T2 = self.condition_points(s2)
            # Compute the homography
            H,HC = self.compute_homography(ps1, ps2, T1, T2)
            # Estamate the distance of homography
            dist = self.compute_homography_distance(H, p1, p2)
            # Get the inliers
            N,inliers = self.find_inliers(pairs, dist, threshold)
            #Update the best inliers and corresbonding homography
            if N > best_N:
                best_N = N
                best_H = np.copy(H)
                best_inliers = np.copy(inliers)
        
        return best_H, best_N, best_inliers
    #
    # You code here
    #


    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
        p1 = inliers[:,0:2]
        p2 = inliers[:,2:4]
        A1 = []
        # Compute the homography based on inliers
        for i in range(0,p1.shape[0]):
            r1 = [0, 0, 0, p1[i][0], p1[i][1], 1, -(p1[i][0]*p2[i][1]), -(p1[i][1]*p2[i][1]), -(p2[i][1])]
            r2 = [-p1[i][0], -p1[i][1], -1 , 0, 0, 0, p1[i][0]*p2[i][0], p1[i][1]*p2[i][0], p2[i][0]]
            A1.append(r1)
            A1.append(r2)
        U, s, V = np.linalg.svd(A1)
        H = V[-1]
        H = H.reshape(3,3)
        Hn = H / H[2][2]
        #print(Hn)
        return Hn
    #
    # You code here
    #