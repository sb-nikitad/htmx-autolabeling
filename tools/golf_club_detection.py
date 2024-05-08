import numpy as np
import copy
import cv2


def inference_2d_golf_club_line_segment(img_idx, ignore_frames, img, results_2d, conf_2d, ref_club, ref_club_length,
                                        ref_direction, ref_img_idx, fps=120):
    def line_association(line_1, line_2):
        '''
        line: [x1, y1, x2, y2]
        '''
        line_1 = np.array(line_1[0])
        line_2 = np.array(line_2[0])

        suppose_direction = line_1[2:] - line_1[:2]
        suppose_direction = suppose_direction / np.linalg.norm(suppose_direction)

        direction1 = line_2[2:] - line_2[:2]
        direction1 = direction1 / np.linalg.norm(direction1)

        direction2 = line_2[:2] - line_1[2:]
        direction2 = direction2 / np.linalg.norm(direction2)

        angle1 = direction1.dot(suppose_direction)
        angle2 = direction2.dot(suppose_direction)
        if 1 - angle1 < 0.0025:
            dist = np.linalg.norm(line_2[:2] - line_1[2:])
            if dist > 10 and 1 - angle2 > 0.0025:  # 4 degree
                return np.inf
            else:
                return dist
        else:
            return np.inf

    def get_foot_point(point, line_p1, line_p2):
        """
        point, line_p1, line_p2 : [x, y]
        """
        x0 = point[0]
        y0 = point[1]

        x1 = line_p1[0]
        y1 = line_p1[1]

        x2 = line_p2[0]
        y2 = line_p2[1]

        k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / \
            ((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.0

        xn = k * (x2 - x1) + x1
        yn = k * (y2 - y1) + y1

        return np.array([xn, yn])

    def calc_angle(direction1, direction2):
        angle = direction1.dot(direction2)
        angle = angle / (np.linalg.norm(direction1) * np.linalg.norm(direction2))

        return angle

    def check_hosel(results_2d, conf_2d):
        start = results_2d[35, :]
        end = results_2d[34, :]
        hosel = results_2d[36, :]

        direction1 = end - start
        direction2 = hosel - start

        angle = calc_angle(direction1, direction2)

        if angle < 0.94:
            conf_2d[36, 0] = 0

        return conf_2d

    if fps == 120:
        ref_dir_interval_thres = 3
    else:
        ref_dir_interval_thres = 6
    
    current_direction = results_2d[34, :] - results_2d[35, :]     # 34: mid hands     35: top of handle
    if np.linalg.norm(current_direction) == 0:
        conf_2d[36, 0] = 0
        return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx
    current_direction = current_direction / np.linalg.norm(current_direction)

    if ref_direction is None:
        suppose_direction = current_direction
    elif img_idx - ref_img_idx > ref_dir_interval_thres:
        angle = ref_direction.dot(current_direction) / \
                (np.linalg.norm(ref_direction) * np.linalg.norm(current_direction))
        if 1 - abs(angle) < 0.065:
            suppose_direction = ref_direction
        else:
            ref_direction = None
            ref_club = None
            ref_club_length = None
            suppose_direction = current_direction
    else:
        suppose_direction = ref_direction

    suppose_direction = suppose_direction / np.linalg.norm(suppose_direction)

    min_edge = min(img.shape[0], img.shape[1])
    scale_factor = min_edge / 720 * 1.0

    # bbox: [up_left_corner, bottom_right_corner]
    if img_idx - ignore_frames == 0 or ref_direction is None:
        bbox = np.array([np.clip(results_2d[35, 0] - current_direction[0] * 50 * scale_factor, 0, img.shape[1]),
                         np.clip(results_2d[35, 1] - current_direction[1] * 50 * scale_factor, 0, img.shape[0]),
                         np.clip(results_2d[35, 0] + current_direction[0] * 250 * scale_factor, 0, img.shape[1]),
                         np.clip(results_2d[35, 1] + current_direction[1] * 250 * scale_factor, 0, img.shape[0])])
        bbox = np.array([max(min(bbox[0], bbox[2]) - 50, 0),
                         max(min(bbox[1], bbox[3]) - 50, 0),
                         min(max(bbox[0], bbox[2]) + 50, img.shape[1]),
                         min(max(bbox[1], bbox[3]) + 50, img.shape[0])])

    else:
        bbox = np.array([ref_club[0],
                         ref_club[1],
                         ref_club[2],
                         ref_club[3]])
        bbox = np.array([max(min(bbox[0], bbox[2]) - 75 * scale_factor, 0),
                         max(min(bbox[1], bbox[3]) - 75 * scale_factor, 0),
                         min(max(bbox[0], bbox[2]) + 75 * scale_factor, img.shape[1]),
                         min(max(bbox[1], bbox[3]) + 75 * scale_factor, img.shape[0])])

    # line segment detection requires GRAY format
    try:
        crop_img = img[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2]), :]
    except:
        conf_2d = check_hosel(results_2d, conf_2d)
        return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx

    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(crop_img)

    if lines is None:
        conf_2d = check_hosel(results_2d, conf_2d)
        return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx

    # Filter lines by directions
    interested_lines = []
    for line in lines:
        np_line = np.array(line[0])

        direction = np_line[2:] - np_line[:2]
        direction = direction / np.linalg.norm(direction)
        angle = direction.dot(suppose_direction)

        if abs(angle) > 0.95:
            if angle < 0:
                np_line = np.array([np_line[2], np_line[3], np_line[0], np_line[1]])

            np_line = np.array([[np_line[0] + bbox[0], np_line[1] + bbox[1],
                                 np_line[2] + bbox[0], np_line[3] + bbox[1]]])

            direction1 = np_line[0][:2] - results_2d[34, :]
            direction1 = direction1 / np.linalg.norm(direction1)
            angle1 = direction1.dot(current_direction)

            if angle1 > 0.93:
                interested_lines.append(np_line)

    # Find lines whose stating point is closest to mid hand
    all_dist_interested_lines = []
    for line in interested_lines:
        np_line = np.array(line[0])
        all_dist_interested_lines.append(np.linalg.norm(np_line[:2] - results_2d[34, :]))
    sorted_idx = np.argsort(all_dist_interested_lines).tolist()
    if len(sorted_idx) > 20:
        sorted_idx = sorted_idx[:20]

    lines = []
    initial_lines_idx = copy.deepcopy(sorted_idx)

    for idx1 in initial_lines_idx:
        if all_dist_interested_lines[idx1] > 100:
            break
        traj_lines_idx = [idx1]
        not_traj_lines_idx = [i for i in range(len(interested_lines))]
        not_traj_lines_idx.remove(idx1)

        while len(not_traj_lines_idx) != 0:
            start_idx = traj_lines_idx[0]
            end_idx = traj_lines_idx[-1]
            line_1 = np.array([[interested_lines[start_idx][0][0], interested_lines[start_idx][0][1],
                                interested_lines[end_idx][0][2], interested_lines[end_idx][0][3]]])

            all_dist = np.full(len(not_traj_lines_idx), np.inf)
            for i in range(len(not_traj_lines_idx)):
                idx2 = not_traj_lines_idx[i]
                line_2 = interested_lines[idx2]
                all_dist[i] = line_association(line_1, line_2)

            min_idx = np.argmin(all_dist)
            if all_dist[min_idx] < 50:
                idx2 = not_traj_lines_idx[min_idx]
                traj_lines_idx.append(idx2)
                not_traj_lines_idx.remove(idx2)
            elif all_dist[min_idx] < 100:
                idx2 = not_traj_lines_idx[min_idx]
                tmp_line = interested_lines[idx2][0]
                if np.linalg.norm(tmp_line[2:] - tmp_line[:2]) > all_dist[min_idx]:
                    traj_lines_idx.append(idx2)
                    not_traj_lines_idx.remove(idx2)
                else:
                    break
            else:
                break

        lines.append(traj_lines_idx)

    if len(lines) == 0:
        # print("This frame has no golf club 0: ", img_idx)
        conf_2d = check_hosel(results_2d, conf_2d)
        return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx

    # Filter by angle and length
    club_line = []
    for i in range(len(lines)):
        traj_lines_idx = lines[i]
        start_idx = traj_lines_idx[0]
        end_idx = traj_lines_idx[-1]
        traj = np.array([interested_lines[start_idx][0][0], interested_lines[start_idx][0][1],
                         interested_lines[end_idx][0][2], interested_lines[end_idx][0][3]])

        club_line.append(traj)

    length = np.zeros(len(club_line))
    for i in range(len(club_line)):
        line = club_line[i]
        length[i] = np.linalg.norm(line[2:] - line[:2])

    max_length_idx = np.argsort(length)[-1]
    club_line = np.array(club_line[max_length_idx])

    # Final filter
    club_length = np.linalg.norm(club_line[2:] - club_line[:2])
    if img_idx - ignore_frames == 0:
        ref_club_length = club_length

    if ref_club_length != None:
        if club_length < ref_club_length * 0.6:
            club_line = np.zeros((1, 4))

    if np.all(club_line == 0):
        # print("This frame has no golf club 1: ", img_idx)
        conf_2d = check_hosel(results_2d, conf_2d)
        pass
    else:
        # Compare the results from LPN and LSD, and merge them together
        lpn_hosel = copy.deepcopy(results_2d[36, :])
        lsd_start = copy.deepcopy(club_line[:2])
        lsd_end = copy.deepcopy(club_line[2:])
        foot_point = get_foot_point(lpn_hosel, lsd_start, lsd_end)

        direction1 = lsd_end - lsd_start
        direction2 = lpn_hosel - lsd_start

        angle = calc_angle(direction1, direction2)
        if angle < 0.94:
            results_2d[36, :] = lsd_end
            conf_2d[36, 0] = 1
        else:
            results_2d[36, :] = foot_point

        ref_club = np.append(results_2d[35, :], results_2d[36, :])
        ref_club_length = club_length
        ref_direction = results_2d[36, :] - results_2d[35, :]
        ref_direction = ref_direction / np.linalg.norm(ref_direction)
        ref_img_idx = img_idx

    return results_2d, conf_2d, ref_club, ref_club_length, ref_direction, ref_img_idx