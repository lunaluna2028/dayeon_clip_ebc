import numpy as np

def split_into_blocks(image, labels, block_size):
    # 이미지와 라벨을 블록 단위로 분할
    h, w, _ = image.shape
    blocks = []
    block_labels = []
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append(block)
            
            count = np.sum(
                (labels[:, 0] >= x) & (labels[:, 0] < x + block_size) &
                (labels[:, 1] >= y) & (labels[:, 1] < y + block_size)
            )
            block_labels.append(count)
    
    return blocks, block_labels
