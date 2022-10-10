#  open image.jpg and add noise

import cv2
import numpy as np
import matplotlib.pyplot as plt

# read image
img = cv2.imread('image.jpg')


# show this image in grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Apply random noise to the grayscale image
noise = np.zeros(gray.shape, np.uint8)
cv2.randn(noise, 0, 100)
noisy = cv2.add(gray, noise)


# apply smothing spatial filter to gray and noisy in a 3x3, 7x7, 9x9, 11x11 matrix
blurGray3 = cv2.blur(gray, (3, 3))
blurGray7 = cv2.blur(gray, (7, 7))
blurGray9 = cv2.blur(gray, (9, 9))
blurGray11 = cv2.blur(gray, (11, 11))

blurNoise3 = cv2.blur(noisy, (3, 3))
blurNoise7 = cv2.blur(noisy, (7, 7))
blurNoise9 = cv2.blur(noisy, (9, 9))
blurNoise11 = cv2.blur(noisy, (11, 11))

# apply binomial smoothing filter
blurImgBinomial3 = cv2.blur(gray, (3, 3), cv2.BORDER_DEFAULT)
blurImgBinomial7 = cv2.blur(gray, (7, 7), cv2.BORDER_DEFAULT)
blurImgBinomial9 = cv2.blur(gray, (9, 9), cv2.BORDER_DEFAULT)
blurImgBinomial11 = cv2.blur(gray, (11, 11), cv2.BORDER_DEFAULT)

blurNoiseBinomial3 = cv2.blur(noisy, (3, 3), cv2.BORDER_DEFAULT)
blurNoiseBinomial7 = cv2.blur(noisy, (7, 7), cv2.BORDER_DEFAULT)
blurNoiseBinomial9 = cv2.blur(noisy, (9, 9), cv2.BORDER_DEFAULT)
blurNoiseBinomial11 = cv2.blur(noisy, (11, 11), cv2.BORDER_DEFAULT)

# to gray and noisy apply block detector filter [1, -1]:
blockDetectorGray = cv2.filter2D(gray, cv2.CV_64F, np.array([1, -1]))
blockDetectorNoisy = cv2.filter2D(noisy,  cv2.CV_64F, np.array([1, -1]))

# to gray and noisy apply Prewitt in X:
prewittXGray = cv2.filter2D(
    gray, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
prewittXNoisy = cv2.filter2D(
    noisy, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))

# to gray and noisy apply Prewitt in Y:
prewittYGray = cv2.filter2D(
    gray, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
prewittYNoisy = cv2.filter2D(
    noisy, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

# to gray and noisy apply Sobel in X:
sobelXGray = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobelXNoisy = cv2.Sobel(noisy, cv2.CV_64F, 1, 0, ksize=5)


# to gray and noisy apply Sobel in Y:
sobelYGray = cv2.filter2D(
    gray, cv2.CV_64F, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
sobelYNoisy = cv2.filter2D(
    noisy, cv2.CV_64F, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

# to gray and noisy apply first derivative of Gaussian using a 5x5 matrix:
gaussian1DGray = cv2.filter2D(
    gray, cv2.CV_64F, np.array([[-1, -2, 0, 2, 1], [-4, -8, 0, 8, 4], [-6, -12, 0, 12, 6], [-4, -8, 0, 8, 4], [-1, -2, 0, 2, 1]]))
gaussian1DNoisy = cv2.filter2D(
    noisy, cv2.CV_64F, np.array([[-1, -2, 0, 2, 1], [-4, -8, 0, 8, 4], [-6, -12, 0, 12, 6], [-4, -8, 0, 8, 4], [-1, -2, 0, 2, 1]]))

# to gray and noisy laplacian in a 3x3 matrix:
laplacianGray = cv2.filter2D(
    gray, cv2.CV_64F, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))
laplacianNoisy = cv2.filter2D(
    noisy, cv2.CV_64F, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))

# to gray and noisy apply second derivative of Gaussian using a 5x5, 7x7, 11x11 matrix:
gaussian2DGray5 = cv2.filter2D(
    gray, cv2.CV_64F, np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]))
gaussian2DGray7 = cv2.filter2D(
    gray, cv2.CV_64F, np.array([[0, 0, 0, -1, 0, 0, 0], [0, 0, -1, -2, -1, 0, 0], [0, -1, -2, -4, -2, -1, 0],
                               [-1, -2, -4, 52, -4, -2, -1], [0, -1, -2, -4, -2, -1, 0], [0, 0, -1, -2, -1, 0, 0], [0, 0, 0, -1, 0, 0, 0]])
)
gaussian2DGray11 = cv2.filter2D(
    gray, cv2.CV_64F, np.array([[0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, -2, -1, 0, 0, 0, 0, 0], [0, 0, -1, -2, -4, -2, -1, 0, 0, 0, 0], [0, -1, -2, -4, -8, -4, -2, -1, 0, 0, 0], [-1, -2, -4, -8, 144, -8, -4, -2, -1, 0, 0], [
                               0, -1, -2, -4, -8, -4, -2, -1, 0, 0, 0], [0, 0, -1, -2, -4, -2, -1, 0, 0, 0, 0], [0, 0, 0, -1, -2, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
)

gaussian2DNoisy5 = cv2.filter2D(
    noisy, cv2.CV_64F, np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]))
gaussian2DNoisy7 = cv2.filter2D(
    noisy, cv2.CV_64F, np.array([[0, 0, 0, -1, 0, 0, 0], [0, 0, -1, -2, -1, 0, 0], [0, -1, -2, -4, -2, -1, 0],
                                [-1, -2, -4, 52, -4, -2, -1], [0, -1, -2, -4, -2, -1, 0], [0, 0, -1, -2, -1, 0, 0], [0, 0, 0, -1, 0, 0, 0]])
)
gaussian2DNoisy11 = cv2.filter2D(
    noisy, cv2.CV_64F, np.array([[0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, -2, -1, 0, 0, 0, 0, 0], [0, 0, -1, -2, -4, -2, -1, 0, 0, 0, 0], [0, -1, -2, -4, -8, -4, -2, -1, 0, 0, 0], [-1, -2, -4, -8, 144, -8, -4, -2, -1, 0, 0], [
                                0, -1, -2, -4, -8, -4, -2, -1, 0, 0, 0], [0, 0, -1, -2, -4, -2, -1, 0, 0, 0, 0], [0, 0, 0, -1, -2, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
)

#  to Gray and Nosie apply a blur of matriz 5x5:
blurGray5 = cv2.blur(gray, (5, 5))
blurNoise5 = cv2.blur(noisy, (5, 5))
#  6
#  to Gray, Naise BlurGray5 and BlurNoise5 apply unsharp masking:
unsharpMaskingGray5 = cv2.filter2D(
    gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

unsharpMaskingBlurGray5 = cv2.filter2D(
    blurGray5, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

unsharpMaskingNoise5 = cv2.filter2D(
    noise, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

unsharpMaskingBlurNoise5 = cv2.filter2D(
    blurNoise5, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
# 6 A
unsharpMaskingGray3 = cv2.filter2D(
    blurGray3, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

unsharpMaskingGray7 = cv2.filter2D(
    blurGray7, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
# 6 B
unsharpMaskingBinomialGray3 = cv2.filter2D(
    blurImgBinomial3, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

unsharpMaskingBinomialGray7 = cv2.filter2D(
    blurImgBinomial7, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

#  to blurGray3 apply Discrete Transform Fourier to be show using matplotlib:
dft = cv2.dft(np.float32(blurGray3), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitudeSpectrumGray3 = 20 * \
    np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

#  to blurGray11 apply Discrete Transform Fourier to be show using matplotlib:
dft = cv2.dft(np.float32(blurGray11), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitudeSpectrumGray11 = 20 * \
    np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

#  to blurNoise3 apply Discrete Transform Fourier to be show using matplotlib:
dft = cv2.dft(np.float32(blurNoise3), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitudeSpectrumNoise3 = 20 * \
    np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

#  to blurNoise11 apply Discrete Transform Fourier to be show using matplotlib:
dft = cv2.dft(np.float32(blurNoise11), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitudeSpectrumNoise11 = 20 * \
    np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


# # Show 6 images in a 3x3 grid without spacing
plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1), plt.imshow(gray, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2), plt.imshow(noisy, cmap='gray')
plt.title('Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3), plt.imshow(blurGray3, cmap='gray')
plt.title('Original \nBlur 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(blurGray5, cmap='gray')
plt.title('Original \nBlur 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(blurGray7, cmap='gray')
plt.title('Original \nBlur 7x7'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(blurGray11, cmap='gray')
plt.title('Original \nBlur 11x11'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(blurNoise3, cmap='gray')
plt.title('Blur Ruido \n3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(blurNoise5, cmap='gray')
plt.title('Blur Ruido \n5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(blurNoise7, cmap='gray')
plt.title('Blur Ruido \n7x7'), plt.xticks([]), plt.yticks([])


plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1), plt.imshow(blurNoise9, cmap='gray')
plt.title('Blur Ruido \n9x9'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2), plt.imshow(blurNoise11, cmap='gray')
plt.title('Blur Ruido \n11x11'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3), plt.imshow(blurImgBinomial3, cmap='gray')
plt.title('Original \nBinomial 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(blurImgBinomial7, cmap='gray')
plt.title('Original \nBinomial 7x7'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(blurImgBinomial9, cmap='gray')
plt.title('Original \nBinomial 9x9'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(blurImgBinomial11, cmap='gray')
plt.title('Original \nBinomial 11x11'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(blurNoiseBinomial3, cmap='gray')
plt.title('Ruido \nBinomial 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(blurNoiseBinomial7, cmap='gray')
plt.title('Ruido \nBinomial 7x7'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(blurNoiseBinomial9, cmap='gray')
plt.title('Ruido \nBinomial 9x9'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1), plt.imshow(blurNoiseBinomial11, cmap='gray')
plt.title('Ruido \nBinomial 11x11'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 3, 2), plt.imshow(blockDetectorGray, cmap='gray')
plt.title(
    'Detector de\n Bloques [1 -1]\n original'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3), plt.imshow(blockDetectorNoisy, cmap='gray')
plt.title(
    'Detector de\n Bloques [1 -1]\n ruido'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(prewittXGray, cmap='gray')
plt.title('Prewitt \nX original'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(prewittXNoisy, cmap='gray')
plt.title('Prewitt \nX ruido'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(prewittYGray, cmap='gray')
plt.title('Prewitt \nY original'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(prewittYNoisy, cmap='gray')
plt.title('Prewitt \nY ruido'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(sobelXGray, cmap='gray')
plt.title('Sobel X \noriginal'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(sobelXNoisy, cmap='gray')
plt.title('Sobel X \nruido'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1), plt.imshow(sobelYGray, cmap='gray')
plt.title('Sobel Y \noriginal'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2), plt.imshow(sobelYNoisy, cmap='gray')
plt.title('Sobel Y \nruido'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3), plt.imshow(gaussian1DGray, cmap='gray')
plt.title('Gaussiano \n1D original'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(gaussian1DNoisy, cmap='gray')
plt.title('Gaussiano \n1D ruido'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(laplacianGray, cmap='gray')
plt.title('Laplaciano \noriginal'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(laplacianNoisy, cmap='gray')
plt.title('Laplaciano \nruido'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(gaussian2DGray5, cmap='gray')
plt.title('Gaussiano 2D \noriginal 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(gaussian2DGray7, cmap='gray')
plt.title('Gaussiano 2D \noriginal 7x7'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(gaussian2DGray11, cmap='gray')
plt.title('Gaussiano 2D \noriginal 11x11'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1), plt.imshow(gaussian2DNoisy5, cmap='gray')
plt.title('Gaussiano 2D \nruido 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2), plt.imshow(gaussian2DNoisy7, cmap='gray')
plt.title('Gaussiano 2D \nruido 7x7'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3), plt.imshow(gaussian2DNoisy11, cmap='gray')
plt.title('Gaussiano 2D \nruido 11x11'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(unsharpMaskingGray5, cmap='gray')
plt.title('Unsharp \nMasking \noriginal 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(unsharpMaskingBlurGray5, cmap='gray')
plt.title('Unsharp \nMasking \nblur+Ruido 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(unsharpMaskingNoise5, cmap='gray')
plt.title('Unsharp \nMasking \nruido 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(unsharpMaskingBlurNoise5, cmap='gray')
plt.title('Unsharp \nMasking \nblur+Ruido 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(unsharpMaskingGray3, cmap='gray')
plt.title('Unsharp \nMasking \noriginal 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(unsharpMaskingGray7, cmap='gray')
plt.title('Unsharp \nMasking \noriginal 7x7'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1), plt.imshow(unsharpMaskingBinomialGray3, cmap='gray')
plt.title('Unsharp \nMasking \nbinomial 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 2), plt.imshow(unsharpMaskingBinomialGray7, cmap='gray')
plt.title('Unsharp \nMasking \nbinomial 7x7'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 3), plt.imshow(magnitudeSpectrumGray3, cmap='gray')
plt.title('SpectralMagnitude \n 2.- 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 4), plt.imshow(magnitudeSpectrumGray11, cmap='gray')
plt.title('SpectralMagnitude \n 2.- 11x11'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(magnitudeSpectrumNoise3, cmap='gray')
plt.title('SpectralMagnitude \n 2.- ruido 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(magnitudeSpectrumNoise11, cmap='gray')
plt.title('SpectralMagnitude \n 2.- ruido 11x11'), plt.xticks([]), plt.yticks([])


plt.tight_layout()
plt.show()
