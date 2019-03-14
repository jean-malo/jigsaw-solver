import matplotlib
import networkx as nx
import numpy as np
from PIL import Image
matplotlib.use('TkAgg')
from math import sqrt

class Solver():
    def __init__(self, parent, logger, colorSpace, piece_w, piece_h, path, shuffle, displayInitial, displayEnd,testing):
        super().__init__()
        if not testing:
            parent.saveSolution.setEnabled(False)
        self.parent = parent
        self.testing = testing
        self.colorSpace = colorSpace
        self.piece_w = piece_w
        self.logger = logger
        self.piece_h = piece_h
        self.path = path
        self.shuffle = shuffle
        self.displayInit = displayInitial
        self.displayEnd = displayEnd
        self.piece_num = np.arange(piece_h*piece_w)
        img, height, width = self.jpg_image_to_array(self.path)
        shuffle = shuffle
        np_image, og_images = self.build_pieces(img, piece_h, piece_w, shuffle)
        self.shuffled_og_image = self.show_pieces(np_image, piece_h, piece_w, height, width, self.displayInit)
        self.og_image = self.show_pieces(og_images, piece_h, piece_w, height, width, False)
        result_edge = self.edges(np_image, 4)
        self.results = self.compatibility(result_edge)
        results2 = self.treat_results(self.results)
        graphres, graph_not_opt = self.graph_create(self.results)
        res_arr_pos = self.reconstruct(graphres, np_image, self.results)
        coord_arr1 = self.insert_failure(self.clean_results(res_arr_pos), self.results)
        coord_arr2 = self.trim_coords(coord_arr1, self.piece_w*self.piece_h)
        self.neighbor_comparison(np_image, og_images, coord_arr1, self.piece_w, self.piece_h)
        self.direct_comparison(np_image,og_images,coord_arr1, piece_h*piece_w)
        self.result_image = self.show_pieces_end(self.clean_results(coord_arr2), np_image,self.displayEnd)
        if not testing:
            parent.saveSolution.setEnabled(True)

    #Build the pieces given an image and a number of pieces
    def build_pieces(self, image, piece_num_h, piece_num_w, shuffle_pieces):
        piece_num = piece_num_h*piece_num_w
        images = [None for _ in range(piece_num)]
        width = len(image[0,:])
        height = len(image[:,0])
        split_height = height // piece_num_h
        split_width = width // piece_num_w
        y=0
        for i in range(piece_num_h):
            x = 0
            for j in range(piece_num_w):
                temp_img = image[y:y + split_height, x:x + split_width, :]
                images[i*piece_num_w+j] = temp_img
                x += split_width
            y += split_height
        original_image = images
        if shuffle_pieces:
            images = np.random.permutation(images)
        return images, original_image

    #Builds 4 edges for each image in np_image
    def edges(self, np_image, target) :
        result_edge = [['#' for _ in range(4)] for _ in range(len(np_image))]
        for x in range(0, len(np_image)) :
                north = np_image[x][0:target,:,:]
                south = np_image[x][-target:,:,:]
                west = np_image[x][:, 0:target, :]
                east = np_image[x][:, -target:, :]
                result_edge[x][0] = north
                result_edge[x][1] = east
                result_edge[x][2] = south
                result_edge[x][3] = west
        return result_edge

    #Insert a piece in the grid. Checks for conflict and if at the edge.
    def insert_piece(self, y,x,z,arr, x_tar, results):
        piece_num = len(arr)
        coord = x_tar
        arr_copy = arr
        fail = False
        change = False
        fail_val = -1
        yy = y
        xx = x
        if z == 0:
            if self.at_edge(y,x,z, piece_num):
                arr = self.move_down(arr)
                arr[y,x] = coord
            elif arr[y-1,x] == -1:
                arr[y-1,x] = coord
            else:
                fail = True
                if results[arr[y, x],0,arr[y-1, x]] < results[arr[y, x],0,coord]:
                    fail_val = arr[y-1, x]
                else:
                    change = True
                    fail_val = arr[y-1, x]
                    arr[y-1, x] = coord
        if z == 1:
            if self.at_edge(y,x,z, piece_num):
                arr = self.move_left(arr)
                arr[y,x] = coord
            elif arr[y,x+1] == -1:
                arr[y,x+1] = coord
            else:
                fail = True
                if results[arr[y, x],1,arr[y, x+1]] < results[arr[y, x],1,coord]:
                    fail_val = arr[y, x+1]
                else:
                    change = True
                    fail_val = arr[y, x+1]
                    arr[y, x + 1] = coord
        if z == 2:
            if self.at_edge(y,x,z, piece_num):
                arr = self.move_up(arr)
                arr[y,x] = coord
            elif arr[y+1,x] == -1:
                arr[y+1,x] = coord
            else:
                fail = True
                if results[arr[y, x],2,arr[y+1, x]] < results[arr[y, x],2,coord]:
                    fail_val = arr[y+1, x]
                else:
                    change = True
                    fail_val = arr[y+1, x]
                    arr[y+1, x] = coord
        if z == 3:
            if self.at_edge(y,x,z,piece_num):
                arr = self.move_right(arr)
                arr[y,x] = coord
            elif arr[y,x-1] == -1:
                arr[y,x-1] = coord
            else:
                fail = True
                if results[arr[y, x],3,arr[y, x-1]] < results[arr[y, x],3,coord]:
                    fail_val = arr[y, x-1]
                else:
                    change = True
                    fail_val = arr[y, x-1]
                    arr[y, x - 1] = coord
        if fail:
            return arr, fail_val, fail,change
        else:
            return arr, coord,fail,change

    #Return True or False if the given coordinates are at the edge of the grid.
    def at_edge(self,y,x,dir, piece_num):
        if 0 > (x or y) or (x or y) > piece_num:
            return -1
        if dir == 0:
            return y-1 < 0
        if dir == 2:
            return y+1 > piece_num
        if dir == 1:
            return x+1 > piece_num
        if dir == 3:
            return x-1 < 0

    def move_down(self,arr):
        return np.roll(arr,1, axis=0)

    def move_up(self,arr):
        return np.roll(arr,-1, axis=0)

    def move_right(self,arr):
        return np.roll(arr,1, axis=1)

    def move_left(self,arr):
        return np.roll(arr,-1, axis=1)

    def compatibility(self,np_pieces):
        shape_results = (len(np_pieces),4,len(np_pieces))
        results = np.empty(shape_results)
        results.fill(np.nan)
        for x in range(0, len(np_pieces)):
                for z in range(4):
                    for xx in range(0, len(np_pieces)):
                            if x != xx:
                                zz = self.getInverse(z)
                                results[x][z][xx] = self.get_scores(np_pieces[x][z], np_pieces[xx][zz], z)
                if not self.testing:
                    self.logger.debug(str('Piece ' + str(x) + ' completed...'))

        return results

    #Return the relevant edge position given a position
    def getInverse(self,p):
        if p == 0:
            return 2
        if p == 1:
            return 3
        if p == 2:
            return 0
        if p == 3:
            return 1

    def get_cost(self,p1,p2,results):
        idx = np.argmin([results[p1][z][p2]for z in range(4)])
        return idx, results[p1][idx][p2]

    """
    calculates the compatbility score between two edges
    @return sum of the score p1p2 + p2p1
    """
    def get_scores(self, piece1, piece2, edge):
        if self.colorSpace == 'HSV':
            piece1 = matplotlib.colors.rgb_to_hsv(piece1 / float(256))
            piece2 = matplotlib.colors.rgb_to_hsv(piece2 / float(256))
        else:
            piece1 = piece1 / float(256)
            piece2 = piece2 / float(256)
        dummy_gradients = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, -1, 0],[0, 0, -1]]
        if edge == 0:
            grad_p1 = abs(piece1[0, :, :] - piece1[1, :, :])
            grad_p2 = abs(piece2[-1, :, :] - piece2[-2, :, :])
            grad_p1p2 = abs(piece2[-1, :, :] - piece1[0, :, :])

        elif edge == 1:
            grad_p1 = abs(piece1[:,-1,:] - piece1[:,-2,:])
            grad_p2 = abs(piece2[:, 0, :] - piece2[:, 1, :])
            grad_p1p2 = abs(piece2[:, 0, :] - piece1[:, -1, :])


        elif edge == 2:
            grad_p1 = abs(piece1[-1, :, :] - piece1[-2, :, :])
            grad_p2 = abs(piece2[0, :, :] - piece2[1, :, :])
            grad_p1p2 = abs(piece2[0, :, :] - piece1[-1, :, :])


        elif edge == 3:
            grad_p1 = abs(piece1[:,0, :] - piece1[:, 1, :])
            grad_p2 = abs(piece2[:, -1, :] - piece2[:, -2, :])
            grad_p1p2 = abs(piece2[:, -1, :] - piece1[:, 0, :])

        else:
            raise ValueError('Edge number out of range')

        gr_p1_mean = np.mean(grad_p1)
        gr_p2_mean = np.mean(grad_p2)

        gr_diff_p1_mean = abs(grad_p1p2 - gr_p1_mean)
        gr_diff_p2_mean = abs(grad_p1p2 - gr_p2_mean)

        grad_p1_dummy = np.append(grad_p1, dummy_gradients, axis=0)
        grad_p2_dummy = np.append(grad_p2, dummy_gradients, axis=0)

        #p1_cov = np.cov(grad_p1_dummy, rowvar=False)
        #p2_cov = np.cov(grad_p2_dummy, rowvar=False)

        p1_cov = [[1,0,0],[0,1,0],[0,0,1]]
        p2_cov = [[1,0,0],[0,1,0],[0,0,1]]

        p1_cov_inv = np.linalg.inv(p1_cov)
        p2_cov_inv = np.linalg.inv(p2_cov)


        mahalanobis_distp1p2 = sqrt(np.sum(np.dot(np.dot(gr_diff_p1_mean, p1_cov_inv),np.transpose(gr_diff_p1_mean))))
        mahalanobis_distp2p1 = sqrt(np.sum(np.dot(np.dot(gr_diff_p2_mean, p2_cov_inv),np.transpose(gr_diff_p2_mean))))

        return(mahalanobis_distp1p2 + mahalanobis_distp2p1)

    #Find the best start for Kruskal's algorithm
    def find_best_start(self,graph):
        start_node = (n for n in graph if len(list(graph.neighbors(n))) == 1)
        cur_length = 10000
        for node in list(start_node):
            new_len = len(list(nx.single_source_shortest_path_length(graph, node, cutoff=3)))
            if new_len < cur_length:
                cur_length = new_len
                target_node = node
        return target_node


    #check if move is possible up down left or right (none?) if so do it else move n to left right top or bottom and insert nbr in place of n
    def reconstruct(self,graph, images, results):
        node_num = len(nx.nodes(graph))
        arr = np.zeros((node_num, node_num), dtype=int)
        arr.fill(-1)
        edge_done = []
        start_node_tup = self.find_best_start(graph)
        tt = list(nx.bfs_successors(graph, start_node_tup))
        arr_list = []
        edge_not_done = []
        for x, node_list in enumerate(tt):
            curr_node = node_list[0]
            if len(edge_done) == 0:
                arr[0, 0] = curr_node
                edge_done.append([curr_node,curr_node])
                x = 0
                y = 0
            else:
                res = np.transpose(np.where(arr == [x for x in edge_done if x[0] == curr_node][0][1]))
                x = res[0,1]
                y = res[0,0]
            for nodes in node_list:
                if np.any(nodes != curr_node):
                    for node in nodes:
                        target_node = node
                        p1_data = graph[curr_node][target_node]['p1']
                        p2_data = graph[curr_node][target_node]['p2']
                        if p1_data[0] == curr_node:
                            edge = p1_data[-1]
                        else:
                            edge = p2_data[-1]
                        arr, val_arr,fail,change = self.insert_piece(y, x, edge, arr, target_node, results)
                        #arr, val_arr= insert_piece(y, x, edge, arr, target_node, results)
                        if fail:
                            if change:
                                edge_done = self.replace_val(edge_done, val_arr, target_node)
                                val_arr = target_node
                        edge_done.append([target_node, val_arr])
        return arr

    def replace_val(self,list,val,target):
        for x in list:
            if x[0] == val:
                x[1] = target
        return list

    def clean_results(self,arr):
        arr = arr[~np.all(arr == -1, axis=1)]
        arr = np.transpose(arr)
        arr = arr[~np.all(arr == -1, axis=1)]
        arr = np.transpose(arr)
        return arr

    def treat_results(self,results):
        for x, val in enumerate(results):
            for z in range(4):
                idx = np.argsort(results[x][z])[:2]
                results[x][z] = np.divide(results[x][z], results[x][z][idx[1]])
        return results

    def get_score3(self, a, b):
        return np.amin([self.results[a][z][b] for z in range(4)])

    def get_best_match(self,piece_X, results, edge):
        piece_result = results[piece_X][edge]
        idx = np.argsort(piece_result)[:2]
        x = idx[0]
        xx= idx[1]
        score = piece_result[x] / piece_result[xx]
        return (x,score)

    #Creates a graph. Return the minimum spanning tree and the graph
    def graph_create(self,results):
        graph = nx.Graph()
        for x in range(0, len(results)):
                for z in range(4):
                    match = self.get_best_match(x,results,z)
                    match_x = match[0]
                    match_score = match[1]
                    #match_score = results[x][z][match_x]
                    if np.isnan(match_score):
                        match_score = 100000
                    # Check if edge already exists, if it does only replace current edge if match_score is smaller than existing score (this can happen at edges)
                    if not(graph.get_edge_data(x,match_x) is not None and graph.get_edge_data(x,match_x)['weight'] > match_score):
                        graph.add_edge(x, match_x, weight=match_score, p1=(x,z), p2=(match_x,self.getInverse(z)))
        T = nx.minimum_spanning_tree(graph)
        return T, graph





    def trim_coords(self,coord, piece_num):
        piece_number_list = np.arange(piece_num)
        top_holes = np.transpose(np.where(coord[0,:] == -1))
        top_holes_percent = len(top_holes) / len(coord[0,:])
        bottom_holes = np.transpose(np.where(coord[-1, :] == -1))
        bottom_holes_percent  = len(bottom_holes) / len(coord[-1, :])
        left_holes = np.transpose(np.where(coord[:, 0] == -1))
        left_holes_percent  = len(left_holes) / len(coord[:, 0])
        right_holes = np.transpose(np.where(coord[:, -1] == -1))
        right_holes_percent  = len(right_holes) / len(coord[:, -1])
        holes = [top_holes_percent, right_holes_percent, bottom_holes_percent,left_holes_percent ]
        largest_holes = holes.index(max(holes))
        if max(holes) == 0:
            return coord
        if largest_holes == 0:
            #fails.extend(coord[0,np.where(coord[0,:]!= -1)].ravel())
            coord = coord[1:,:]
        if largest_holes == 1:
            #fails.extend(coord[np.where(coord[:,-1]!= -1),-1].ravel())
            coord = coord[:,:-1]
        if largest_holes == 2:
            #fails.extend(coord[-1,np.where(coord[-1,:]!= -1)].ravel())
            coord = coord[:-1,:]
        if largest_holes == 3:
            #fails.extend(coord[np.where(coord[0,:]!= -1),0].ravel())
            coord = coord[:,1:]
        fails = list(piece_number_list[np.logical_not(np.in1d(piece_number_list, coord))])
        pos_l = np.transpose(np.where(coord == -1))

        if len(np.transpose(np.where(coord == -1))) <= len(fails):
            return coord
        else:
            return self.trim_coords(coord, piece_num)

    #Inserts the missing pieces. Return an improved version of the coordinate array.
    def insert_failure(self, coord_arr, results):
        for i in range(20):
            fails = list(set(self.piece_num) - set(coord_arr.ravel()))
            for fail in fails:
                fail_edge0 = np.nanargmin(results[fail,0])
                fail_edge1 = np.nanargmin(results[fail,1])
                fail_edge2 = np.nanargmin(results[fail,2])
                fail_edge3 = np.nanargmin(results[fail,3])
                position_0 = np.transpose(np.where(coord_arr == fail_edge0)).ravel()
                position_1 = np.transpose(np.where(coord_arr == fail_edge1)).ravel()
                position_2 = np.transpose(np.where(coord_arr == fail_edge2)).ravel()
                position_3 = np.transpose(np.where(coord_arr == fail_edge3)).ravel()
                try:
                    if position_1[0] == position_3[0] and (position_1[1]-1 == position_3[1] or position_1[1]-1 == position_3[1]+1):
                        if coord_arr[position_1[0],position_3[1]+1] != -1:
                            fails.append(coord_arr[position_1[0],position_3[1]+1])
                        coord_arr[position_1[0], position_3[1] + 1] = fail
                        fails.remove(fail)
                        continue
                except Exception:
                    print('Position is null')
                try:
                    if position_0[0] == position_3[0] and (position_1[1]-1 == position_3[1] or position_1[1]-1 == position_3[1]+1):
                        if coord_arr[position_1[0],position_3[1]+1] != -1:
                            fails.append(coord_arr[position_1[0],position_3[1]+1])
                        coord_arr[position_1[0], position_3[1] + 1] = fail
                        fails.remove(fail)
                        continue
                except Exception:
                    print('Position is null')
                try:
                    if (position_1[0] == position_2[0] -1) and (position_1[1]-1 == position_2[1]):
                        if coord_arr[position_1[0], position_2[1]] != -1:
                            fails.append(coord_arr[position_1[0], position_2[1]])
                        coord_arr[position_1[0], position_2[1]] = fail
                        fails.remove(fail)
                        continue
                except Exception:
                    print('Position is null')
                try:
                    if (position_0[1] == position_2[1]) and ((position_0[0]+1 == position_2[0]-1) or position_0[0]+1 == position_2[0]or position_0[0]+1 == position_2[0]-2):
                        if coord_arr[position_0[0]+1, position_2[1]] != -1:
                            fails.append(coord_arr[position_0[0]+1, position_2[1]])
                        coord_arr[position_0[0]+1, position_2[1]] = fail
                        fails.remove(fail)
                        continue
                except Exception:
                    print('Position is null')
                try:
                    if (position_0[0] == position_1[0]-1) and (position_0[1] == position_1[1]-1):
                        if coord_arr[position_0[0]+1, position_0[1]] != -1:
                            fails.append(coord_arr[position_0[0]+1, position_0[1]])
                        coord_arr[position_0[0]+1, position_0[1]] = fail
                        fails.remove(fail)
                        continue
                except Exception:
                    print('Position is null')
            return coord_arr

    def show_pieces(self,np_array, piece_num_h, piece_num_w,height,width, display):
        y = 0
        w = len(np_array[0][0,:])
        h = len(np_array[0][:,0])
        result = Image.new('RGB', (w*piece_num_w, h*piece_num_h))
        for i in range(piece_num_h):
            x = 0
            for j in range(piece_num_w):
                id = i*piece_num_w+j
                temp_img = Image.fromarray(np_array[id])
                x_offset = len(np_array[id][0, :])
                y_offset = len(np_array[id][:, 0])
                result.paste(temp_img, (x, y))
                x += x_offset
            y += y_offset
        if display:
            result.show()
        return result

    #Displays the image given coordinates.
    def show_pieces_end(self,coord_arr,images,display):
        y = 0
        num_p = len(images)
        w = len(images[0][0,:])
        h = len(images[0][:,0])
        black = np.zeros((h,w,3),dtype=np.uint8)
        black.fill(1)
        result = Image.new('RGB', (w*num_p,h*num_p))
        for i in range(0,len(coord_arr)):
            x=0
            for j in range(0,len(coord_arr[i])):
                if coord_arr[i][j] == -1:
                    temp_img = Image.fromarray(black)
                    x_offset = len(black[0,:])
                    y_offset = len(black[:,0])
                else:
                    xx = coord_arr[i][j]
                    xx = int(xx)
                    temp_img = Image.fromarray(images[xx])
                    x_offset = len(images[xx][0,:])
                    y_offset = len(images[xx][:,0])
                result.paste(temp_img, (x,y))
                x += x_offset
            y += y_offset
        image_data = np.asarray(result)
        non_empty_columns = np.where(image_data.max(axis=0) > 0)[0]
        non_empty_rows = np.where(image_data.max(axis=1) > 0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        image_data_new = image_data[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]
        new_image = Image.fromarray(image_data_new)
        if display:
            new_image.show()
        return new_image

    def jpg_image_to_array(self,image_path):
      """
      Loads JPEG image into 3D Numpy array of shape
      (width, height, channels)
      """
      with Image.open(image_path) as image:
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
      return im_arr, image.size[1], image.size[0]


    #Performs the direct_comparison measure. Returns the score for the given coordinates.
    def direct_comparison(self, shuffled_piece, original_piece, coordinates, piece_number):
        flat_coord = coordinates.flatten()
        count = 0
        for x, val in enumerate(flat_coord):
            try:
                if np.all(shuffled_piece[val] == original_piece[x]):
                    count += 1
            except:
                print('Comparison failed since piece out of bounds of original solution')
        self.logger.debug(str('RESULT of direct comparison: ' + str(count/piece_number)))

    #Performs the neighbor_comparison measure. Returns the score for the given coordinates.
    def neighbor_comparison(self, shuffled_piece, original_piece, coordinates, piece_w,piece_h):
        shape_coord = coordinates.shape
        list_stacked = np.arange(piece_w*piece_h)
        list_stacked = list_stacked.reshape(piece_h,piece_w)
        w = shape_coord[1]
        h = shape_coord[0]
        counts = []
        for (y,x), val in np.ndenumerate(coordinates):
            cand = 0
            count = 0
            if val != -1:
                tar_val = [np.array_equal(shuffled_piece[val],x) for x in original_piece].index(True)
                yy,xx = np.transpose(np.where(tar_val == list_stacked))[0]
                if y - 1 > 0 and yy -1 > 0:
                    cand += 1
                    idx = np.ravel_multi_index((yy-1,xx),list_stacked.shape)
                    if np.all(shuffled_piece[coordinates[y - 1, x]] == original_piece[idx]):
                        count +=1
                if x + 1 < w and xx +1 < list_stacked.shape[1]:
                    cand += 1
                    idx = np.ravel_multi_index((yy,xx+1),list_stacked.shape)
                    if np.all(shuffled_piece[coordinates[y, x + 1]]== original_piece[idx]):
                        count +=1
                if y + 1 < h and yy +1 < list_stacked.shape[0]:
                    cand += 1
                    idx = np.ravel_multi_index((yy+1,xx),list_stacked.shape)
                    if np.all(shuffled_piece[coordinates[y + 1, x]] == original_piece[idx]):
                        count +=1
                if x - 1 > 0 and xx-1 < list_stacked.shape[0]:
                    cand += 1
                    idx = np.ravel_multi_index((yy,xx-1),list_stacked.shape)
                    if np.all(shuffled_piece[coordinates[y , x-1]] == original_piece[idx]):
                        count +=1
                try:
                    counts.append(count/cand)
                except:
                    counts.append(0)
        counts = np.array(counts)
        self.logger.debug(str('RESULT of Neighbor comparison: ' + str(np.mean(counts))))
        return np.mean(counts)



