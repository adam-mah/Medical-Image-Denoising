import cv2
import numpy

cv2.setUseOptimized(True)

# Parameters initialization
sigma = 25
Threshold_Hard3D = 2.7 * sigma  # Threshold for Hard Thresholding
First_Match_threshold = 2500
Step1_max_matched_cnt = 16
Step1_Blk_Size = 8
Step1_Blk_Step = 3
Step1_Search_Step = 3
Step1_Search_Window = 39

Second_Match_threshold = 400
Step2_max_matched_cnt = 32
Step2_Blk_Size = 8
Step2_Blk_Step = 3
Step2_Search_Step = 3
Step2_Search_Window = 39

Beta_Kaiser = 2.0


def init(img, _blk_size, _Beta_Kaiser):
    m_shape = img.shape
    m_img = numpy.matrix(numpy.zeros(m_shape, dtype=float))
    m_wight = numpy.matrix(numpy.zeros(m_shape, dtype=float))
    K = numpy.matrix(numpy.kaiser(_blk_size, _Beta_Kaiser))
    m_Kaiser = numpy.array(K.T * K)
    return m_img, m_wight, m_Kaiser


def Locate_blk(i, j, blk_step, block_Size, width, height):
    if i * blk_step + block_Size < width:
        point_x = i * blk_step
    else:
        point_x = width - block_Size

    if j * blk_step + block_Size < height:
        point_y = j * blk_step
    else:
        point_y = height - block_Size

    m_blockPoint = numpy.array((point_x, point_y), dtype=int)

    return m_blockPoint


def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    point_x = _BlockPoint[0]
    point_y = _BlockPoint[1]

    # SearchWindow
    LX = point_x + Blk_Size / 2 - _WindowSize / 2
    LY = point_y + Blk_Size / 2 - _WindowSize / 2
    RX = LX + _WindowSize
    RY = LY + _WindowSize

    if LX < 0:
        LX = 0
    elif RX > _noisyImg.shape[0]:
        LX = _noisyImg.shape[0] - _WindowSize
    if LY < 0:
        LY = 0
    elif RY > _noisyImg.shape[0]:
        LY = _noisyImg.shape[0] - _WindowSize

    return numpy.array((LX, LY), dtype=int)


def Step1_fast_match(_noisyImg, _BlockPoint):
    (present_x, present_y) = _BlockPoint
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_img = cv2.dct(img.astype(numpy.float64))

    Final_similar_blocks[0, :, :] = dct_img
    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) / Search_Step
    (present_x, present_y) = Window_location

    '''changing'''
    similar_blocks = numpy.zeros((int(blk_num ** 2), int(Blk_Size), int(Blk_Size)), dtype=float)
    m_Blkpositions = numpy.zeros((int(blk_num ** 2), 2), dtype=int)
    Distances = numpy.zeros(int(blk_num ** 2), dtype=float)

    # Search_Window
    matched_cnt = 0
    for i in range(int(blk_num)):
        for j in range(int(blk_num)):
            tem_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(numpy.float64))
            m_Distance = numpy.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            if m_Distance < Threshold and m_Distance > 0:
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]
    return Final_similar_blocks, blk_positions, Count


def Step1_3DFiltering(_similar_blocks):
    statis_nonzero = 0
    m_Shape = _similar_blocks.shape

    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j])
            tem_Vct_Trans[numpy.abs(tem_Vct_Trans[:]) < Threshold_Hard3D] = 0.
            statis_nonzero += tem_Vct_Trans.nonzero()[0].size
            _similar_blocks[:, i, j] = cv2.idct(tem_Vct_Trans)[0]
    return _similar_blocks, statis_nonzero


def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    _shape = _similar_blocks.shape
    if _nonzero_num < 1:
        _nonzero_num = 1
    block_wight = (1. / _nonzero_num) * Kaiser
    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = (1. / _nonzero_num) * cv2.idct(_similar_blocks[i, :, :]) * Kaiser
        m_basic_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += tem_img
        m_wight_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += block_wight


def BM3D_1st_step(_noisyImg):
    (width, height) = _noisyImg.shape
    block_Size = Step1_Blk_Size
    blk_step = Step1_Blk_Step
    Width_num = (width - block_Size) / blk_step
    Height_num = (height - block_Size) / blk_step

    Basic_img, m_Wight, m_Kaiser = init(_noisyImg, Step1_Blk_Size, Beta_Kaiser)

    for i in range(int(Width_num) + 2):
        for j in range(int(Height_num) + 2):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Positions, Count = Step1_fast_match(_noisyImg, m_blockPoint)
            Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)
            Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)
    Basic_img[:, :] /= m_Wight[:, :]
    basic = numpy.matrix(Basic_img, dtype=int)
    basic.astype(numpy.uint8)

    return basic


def Step2_fast_match(_Basic_img, _noisyImg, _BlockPoint):
    (present_x, present_y) = _BlockPoint
    Blk_Size = Step2_Blk_Size
    Threshold = Second_Match_threshold
    Search_Step = Step2_Search_Step
    max_matched = Step2_max_matched_cnt
    Window_size = Step2_Search_Window

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)
    Final_similar_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)
    Final_noisy_blocks = numpy.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_img = cv2.dct(img.astype(numpy.float32))
    Final_similar_blocks[0, :, :] = dct_img

    n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_n_img = cv2.dct(n_img.astype(numpy.float32))
    Final_noisy_blocks[0, :, :] = dct_n_img

    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) / Search_Step
    (present_x, present_y) = Window_location

    similar_blocks = numpy.zeros((int(blk_num ** 2), int(Blk_Size), int(Blk_Size)), dtype=float)
    m_Blkpositions = numpy.zeros((int(blk_num ** 2), 2), dtype=int)
    Distances = numpy.zeros(int(blk_num ** 2), dtype=float)

    # Search_Window
    matched_cnt = 0
    for i in range(int(blk_num)):
        for j in range(int(blk_num)):
            tem_img = _Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(numpy.float32))
            m_Distance = numpy.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            if m_Distance < Threshold and m_Distance > 0:
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]

            (present_x, present_y) = m_Blkpositions[Sort[i - 1], :]
            n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            Final_noisy_blocks[i, :, :] = cv2.dct(n_img.astype(numpy.float64))

    return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count


def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs):
    m_Shape = _Similar_Bscs.shape
    Wiener_wight = numpy.zeros((m_Shape[1], m_Shape[2]), dtype=float)

    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_vector = _Similar_Bscs[:, i, j]
            tem_Vct_Trans = numpy.matrix(cv2.dct(tem_vector))
            Norm_2 = numpy.float64(tem_Vct_Trans.T * tem_Vct_Trans)
            m_weight = Norm_2 / (Norm_2 + sigma ** 2)
            if m_weight != 0:
                Wiener_wight[i, j] = 1. / (m_weight ** 2 * sigma ** 2)
            # else:
            #     Wiener_wight[i, j] = 10000
            tem_vector = _Similar_Imgs[:, i, j]
            tem_Vct_Trans = m_weight * cv2.dct(tem_vector)
            _Similar_Bscs[:, i, j] = cv2.idct(tem_Vct_Trans)[0]

    return _Similar_Bscs, Wiener_wight


def Aggregation_Wiener(_Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
    _shape = _Similar_Blks.shape
    block_wight = _Wiener_wight  # * Kaiser

    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = _Wiener_wight * cv2.idct(_Similar_Blks[i, :, :])  # * Kaiser
        m_basic_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += tem_img
        m_wight_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += block_wight


def BM3D_2nd_step(_basicImg, _noisyImg):
    (width, height) = _noisyImg.shape
    block_Size = Step2_Blk_Size
    blk_step = Step2_Blk_Step
    Width_num = (width - block_Size) / blk_step
    Height_num = (height - block_Size) / blk_step

    m_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Beta_Kaiser)

    for i in range(int(Width_num) + 2):
        for j in range(int(Height_num) + 2):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(_basicImg, _noisyImg, m_blockPoint)
            Similar_Blks, Wiener_wight = Step2_3DFiltering(Similar_Blks, Similar_Imgs)
            Aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)
    m_img[:, :] /= m_Wight[:, :]
    Final = numpy.matrix(m_img, dtype=int)
    Final.astype(numpy.uint8)

    return Final


def bm3d_denoise(noisy_image):
    noisy_image = noisy_image.reshape(noisy_image.shape[0], 64, 64)
    noisy_image = noisy_image * 255.0
    denoised = []
    for i in range(0,1):
        Basic_img = BM3D_1st_step(noisy_image[i])
        Final_img = BM3D_2nd_step(Basic_img, noisy_image[i])
        denoised.append(Final_img)
        print("Image " + str(i) + " denoised")
    return numpy.array(denoised)
