#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

class hf_NN:
    def __init__(self, n_cell):
        self.n_cell = n_cell
        self.weight = np.zeros((n_cell, n_cell))
        self.shita = 0.1

    def train(self, patterns):
        # initialize weight for repetitive learning?
        for i in range(len(patterns)):
            for j in range(self.n_cell):
                for k in range(self.n_cell):
                    self.weight[j][k] += (patterns[i][j] * patterns[i][k]) / float(len(patterns))
        for l in range(self.n_cell):
            self.weight[l][l] = 0

    def test(self, patterns, noised_patterns):
        recall = []
        similarity = []
        correct_rate = []

        for i in range(len(patterns)):
            pattern = patterns[i].copy()
            noise_pre = noised_patterns[i].copy()
            noise_post = noise_pre.copy()

            diff_const = 0
            d = 0
            while diff_const < self.n_cell * 2:
                diff_const = 0
                cell_array = np.random.randint(0, self.n_cell - 1, self.n_cell * 2)
                for j in cell_array:
                    potential = 0.0
                    for k in range(self.n_cell):
                        potential += float(self.weight[j][k]) * noise_pre[k]
                    noise_post[j] = sign(potential - self.shita)
                    rec = noise_post[j]
                    if rec == noise_pre[j]:
                        diff_const += 1
                    noise_pre[j] = rec

            recalled_pattern = noise_post.copy()
            recall.append(recalled_pattern)

            r_zncc, correct = calc_similarity(pattern, recalled_pattern)

            similarity.append(r_zncc)
            correct_rate.append(correct)

        return recall, similarity, correct_rate

def binarize_pattern(src):
    threshold = 100
    # image should be 1d array
    src = src.astype(np.int)
    src = src.flatten()
    dst = src.copy()
    for i in range(len(src)):
        if src[i] >= threshold:
            dst[i] = 1
        else:
            dst[i] = -1
    return dst

def debinarize_pattern(src):
    dst = src.copy()
    for i in range(len(src)):
        if src[i] >= 0:
            dst[i] = 255
        else:
            dst[i] = 0
    return dst

def noise_pattern(img, noise_rate):
    noised_img = np.zeros(img.shape)
    for i in range(len(noised_img)):
        pixel_noise = np.random.rand()
        if pixel_noise <= noise_rate:
            noised_img[i] = -1 * img[i]
        else:
            noised_img[i] = 1 * img[i]
    return noised_img

def noise_patterns(imgs, noise_rate):
    patterns = []
    for i in range(len(imgs)):
        img = imgs[i].copy()
        noise = noise_pattern(img, noise_rate)
        patterns.append(noise)
    return patterns

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def calc_diff(pre, post):
    diff = 0
    for i in range(len(pre)):
        if pre[i] != post[i]:
            diff += 1
    return diff

def calc_similarity(original, recalled):
    # debinarize as for setup
    #tmp = debinarize_pattern(original)
    #img = debinarize_pattern(recalled)
    tmp = original.copy()
    img = recalled.copy()
    # ZNCC
    tmp = tmp - float(np.mean(tmp))
    img = img - float(np.mean(img))

    numer = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for i in range(len(tmp)):
        numer += float(img[i] * tmp[i])
        denom1 += float(img[i] ** 2)
        denom2 += float(tmp[i] ** 2)
    denom = np.sqrt(denom1 * denom2)

    if denom == 0:
        return 0
    r_zncc = numer / denom

    if r_zncc == 1.0:
        correct = 1.0
    else:
        correct = 0.0

    return r_zncc, correct

if __name__ == '__main__':
    # load patterns
    pattern1_src = cv2.imread('imgs/i.jpg', 0)
    pattern1 = binarize_pattern(pattern1_src)
    pattern2_src = cv2.imread('imgs/ro.jpg', 0)
    pattern2 = binarize_pattern(pattern2_src)
    pattern3_src = cv2.imread('imgs/ha.jpg', 0)
    pattern3 = binarize_pattern(pattern3_src)
    pattern4_src = cv2.imread('imgs/ni.jpg', 0)
    pattern4 = binarize_pattern(pattern4_src)
    pattern5_src = cv2.imread('imgs/ho.jpg', 0)
    pattern5 = binarize_pattern(pattern5_src)
    pattern6_src = cv2.imread('imgs/he.jpg', 0)
    pattern6 = binarize_pattern(pattern6_src)
    pattern7_src = cv2.imread('imgs/to.jpg', 0)
    pattern7 = binarize_pattern(pattern7_src)
    patterns = []
    patterns.append(pattern1)
    patterns.append(pattern2)
    patterns.append(pattern3)
    patterns.append(pattern4)
    patterns.append(pattern5)
    patterns.append(pattern6)
    patterns.append(pattern7)

    # make and train a hopfield neural network
    hopfield_NN = hf_NN(5*5)
    hopfield_NN.train(patterns)

    """
    1. test ability for 2 images
    """
    # before noise
    pat1 = debinarize_pattern(patterns[0])
    pat2 = debinarize_pattern(patterns[1])
    pat1 = np.reshape(pat1, (5, 5))
    pat2 = np.reshape(pat2, (5, 5))
    plt.imshow(pat1, interpolation = 'none')
    plt.gray()
    #plt.show()
    plt.imshow(pat2, interpolation = 'none')
    plt.gray()
    #plt.show()

    # after noise
    noised_patterns = noise_patterns(patterns, 0.25)
    noise1 = debinarize_pattern(noised_patterns[0])
    noise2 = debinarize_pattern(noised_patterns[1])
    noise1 = np.reshape(noise1, (5, 5))
    noise2 = np.reshape(noise2, (5, 5))
    plt.imshow(noise1, interpolation = 'none')
    plt.gray()
    #plt.show()
    plt.imshow(noise2, interpolation = 'none')
    plt.gray()
    #plt.show()

    # recall
    recall, similarity, correct_rate = hopfield_NN.test(patterns, noised_patterns)

    # after recall
    out1 = debinarize_pattern(recall[0])
    out2 = debinarize_pattern(recall[1])
    out1 = np.reshape(out1, (5, 5))
    out2 = np.reshape(out2, (5, 5))
    plt.imshow(out1, interpolation = 'none')
    plt.gray()
    #plt.show()
    plt.imshow(out2, interpolation = 'none')
    plt.gray()
    #plt.show()
    plt.clf()

    """
    2. test ability by increase in number of images
    3. test ability by increase in noise rate
    """
    #num_imgs = [2, 3]
    #num_imgs = [2, 3, 4, 5, 6, 7]
    num_imgs = [2, 4]
    n_rate = []
    sim = []
    cor = []

    #test the hopfield network's ability
    for nr in range(5, 55, 5):
        n_r = float(nr) * 0.01
        n_rate.append(n_r)
        similarities = []
        correct_rates = []
        for i in num_imgs:
            patterns_lim = patterns[:i]
            num_trials = 50
            avg_similarity = np.zeros(i)
            avg_correct_rate = np.zeros(i)
            for j in range(num_trials):
                noised_patterns = noise_patterns(patterns_lim, n_r)
                recall, similarity, correct_rate = hopfield_NN.test(patterns_lim, noised_patterns)
                avg_similarity += similarity
                avg_correct_rate += correct_rate
            avg_similarity /= float(num_trials)
            avg_correct_rate /= float(num_trials)

            print ""
            print n_r
            print "number of patterns : %d" %(i)
            #print "similarity : %f" %(avg_similarity)
            print "similarity : "
            print avg_similarity
            #print "correct_rate : %f" %(avg_correct_rate)
            print "correct_rate : "
            print avg_correct_rate

            #calculate the degree of similarity and correct rate in total
            total_similarity = 0.0
            total_correct_rate = 0.0
            for k in range(i):
                total_similarity += avg_similarity[k]
                total_correct_rate += avg_correct_rate[k]
            total_similarity /= float(i)
            total_correct_rate /= float(i)

            similarities.append(total_similarity)
            correct_rates.append(total_correct_rate)
        sim.append(similarities)
        cor.append(correct_rates)

    print sim
    print cor

    """
    # 2. test ability by increase in number of images
    # plot degree of similarity
    plt.plot(num_imgs, similarities, '-o')
    #plt.legend()
    plt.grid()
    plt.title('degree of similarity')
    plt.xlabel('number of images []')
    plt.ylabel('degree of similarity []')
    plt.ylim(-1, 1)
    plt.show()

    # plot correct rate
    plt.plot(num_imgs, correct_rates, '-o')
    #plt.legend()
    plt.grid()
    plt.title('correct rate ')
    plt.xlabel('number of images []')
    plt.ylabel('correct rate []')
    plt.ylim(0, 1)
    plt.show()
    """

    # 3. test ability by increase in noise rate

    noise_rate_2 = n_rate
    noise_rate_4 = n_rate
    similarity_2 = []
    similarity_4 = []
    correct_rate_2 = []
    correct_rate_4 = []
    for i in range(len(n_rate)):
        similarity_2.append(sim[i][0])
        similarity_4.append(sim[i][1])
        correct_rate_2.append(cor[i][0])
        correct_rate_4.append(cor[i][1])

    # plot degree of similarity
    plt.plot(noise_rate_2, similarity_2, '-o', label = "2 images")
    plt.plot(noise_rate_4, similarity_4, '-o', label = "4 images")
    plt.legend()
    plt.grid()
    plt.title('degree of similarity')
    plt.xlabel('noise rate []')
    plt.ylabel('degree of similarity []')
    plt.ylim(-1, 1)
    plt.show()

    # plot correct rate
    plt.plot(noise_rate_2, correct_rate_2, '-o', label = "2 images")
    plt.plot(noise_rate_4, correct_rate_4, '-o', label = "4 images")
    plt.legend()
    plt.grid()
    plt.title('correct rate')
    plt.xlabel('noise rate []')
    plt.ylabel('correct rate []')
    plt.ylim(0, 1)
    plt.show()
