import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io

def calc_div_fft(Vx, Vy):
    Nx = np.shape(Vx)[-1]; Ny = Nx
    
    kx = np.fft.fftfreq(Nx).reshape(Nx)*Nx / Nx        # reshaping so that ky multiplies pointwise with y-index in 2D
    ky = np.fft.fftfreq(Ny).reshape(Ny,1)*Ny / Ny
    
    kx[Nx//2] = 0
    ky[Ny//2] = 0             # removing nyquist frequency to reduce noise
    # kx = Nx/(4*dx) * (np.exp(1j*kx_)-1)*(np.exp(1j*ky_)+1)
    # ky = Ny/(4*dy) * (np.exp(1j*kx_)+1)*(np.exp(1j*ky_)-1)
    div_V = np.fft.ifftn((np.fft.fftn(Vx) * kx + np.fft.fftn(Vy) * ky) * 1j * 2. * np.pi)

    return np.real(div_V)

def calc_div_fd(Vx,Vy):
    Vx = np.squeeze(Vx)
    Vy = np.squeeze(Vy)
    Vxdx = np.roll(Vx, -1, axis=0) - np.roll(Vx, 1, axis=0) # divide it by step size to make it physical value.
    Vydy = np.roll(Vy, -1, axis=1) - np.roll(Vy, 1, axis=1) # change to spectral based derivation for mech equi. gives close to exact
    div_V = Vxdx + Vydy

    return div_V

def me_violation(pred):
    return np.mean(np.abs(pred))

def norml2(a):
    return np.sqrt(np.matmul(a,a.T))

print(norml2(np.array([1,1])))

N = 10
test = "256_basicFFT"

P22_true = np.load(f"../data/elasto_plastic/{test}/npy_files/output/P22.npy", allow_pickle=True)/1e9
P23_true = np.load(f"../data/elasto_plastic/{test}/npy_files/output/P23.npy", allow_pickle=True)/1e9
P32_true = np.load(f"../data/elasto_plastic/{test}/npy_files/output/P32.npy", allow_pickle=True)/1e9
P33_true = np.load(f"../data/elasto_plastic/{test}/npy_files/output/P33.npy", allow_pickle=True)/1e9

P22_UNet = np.load(f"../UNet/sigmai/experiments/elasto_plastic_256_non_triv/test_{test}/num_results/P22_pred.npy", allow_pickle=True)
P23_UNet = np.load(f"../UNet/sigmai/experiments/elasto_plastic_256_non_triv/test_{test}/num_results/P23_pred.npy", allow_pickle=True)
P32_UNet = np.load(f"../UNet/sigmai/experiments/elasto_plastic_256_non_triv/test_{test}/num_results/P32_pred.npy", allow_pickle=True)
P33_UNet = np.load(f"../UNet/sigmai/experiments/elasto_plastic_256_non_triv/test_{test}/num_results/P33_pred.npy", allow_pickle=True)

# mat_content = scipy.io.loadmat(f'../FNO/micromechanics/pred/nontriv_res_256_mode_20_output_layer/{test}/nontriv_res_256_mode_20_output_layer.mat')     # returns a dictionary 
# pred = mat_content.get('pred')/1e9
# P22_FNO = pred[:,:,:,1]
# P23_FNO = pred[:,:,:,2]
# P32_FNO = pred[:,:,:,3]
# P33_FNO = pred[:,:,:,4]

summed_me_v_UNet = np.zeros(2)
summed_me_v_FNO = np.zeros(2)
summed_me_v_spectral = np.zeros(2)

for id_case in range(N):

    Vy1 = P22_true[id_case][np.newaxis,:,:]
    Vz1 = P23_true[id_case][np.newaxis,:,:]
    Vy2 = P32_true[id_case][np.newaxis,:,:]
    Vz2 = P33_true[id_case][np.newaxis,:,:]
    # Vy = P32[900+id_case]
    # Vz = P33[900+id_case]
    divP_true1 = calc_div_fft(Vy1, Vz1)
    divP_true2 = calc_div_fft(Vy2, Vz2)

    # predictions from U-Net
    Vy1 = P22_UNet[id_case][np.newaxis,:,:]
    Vz1 = P23_UNet[id_case][np.newaxis,:,:]
    Vy2 = P32_UNet[id_case][np.newaxis,:,:]
    Vz2 = P33_UNet[id_case][np.newaxis,:,:]
    # Vy = P32[id_case]
    # Vz = P33[id_case]
    divP_UNet1 = calc_div_fft(Vy1, Vz1)
    divP_UNet2 = calc_div_fft(Vy2, Vz2)

    ## predictions from FNO
    # Vy1 = P22_FNO[id_case][np.newaxis,:,:]
    # Vz1 = P23_FNO[id_case][np.newaxis,:,:]
    # Vy2 = P32_FNO[id_case][np.newaxis,:,:]
    # Vz2 = P33_FNO[id_case][np.newaxis,:,:]
    # # Vy = P32[id_case]
    # # Vz = P33[id_case]
    # divP_FNO1 = calc_div_fft(Vy1, Vz1)
    # divP_FNO2 = calc_div_fft(Vy2, Vz2)

    summed_me_v_UNet += [me_violation(divP_UNet1),me_violation(divP_UNet2)]
    # summed_me_v_FNO  += [me_violation(divP_FNO1),me_violation(divP_FNO2)]
    summed_me_v_spectral  += [me_violation(divP_true1),me_violation(divP_true2)]


print(f"Mean absolute me violation for {N} {test} cases - \nUNet - ", norml2(summed_me_v_UNet/N))
print("FNO - ", norml2(summed_me_v_FNO/N))
print("Spectral - ", norml2(summed_me_v_spectral/N))

print("For last case, we print images of divP and following data - ")
print(f"True: Real part (min,max,mean-abs-value): \t{np.min(np.real(divP_true1))}\t{np.max(np.real(divP_true1))}\t{np.mean(np.abs(np.real(divP_true1)))}")
imdata = np.squeeze(np.real(divP_true1))
vmin=np.min(imdata)
vmax=np.max(imdata)
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
plt.figure()
plt.set_cmap("seismic")
plt.imshow(imdata,norm=norm)
plt.colorbar()
plt.savefig(f"divp_true.png")
plt.close()

print(f"Pred: Real part (min,max,mean-abs-value): \t{np.min(np.real(divP_UNet1))}\t{np.max(np.real(divP_UNet1))}\t{np.mean(np.abs(np.real(divP_UNet1)))}")
imdata = np.squeeze(np.real(divP_UNet1))
vmin=np.min(imdata)
vmax=np.max(imdata)
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
plt.figure()
plt.set_cmap("seismic")
plt.imshow(imdata,norm=norm)
plt.colorbar()
plt.savefig(f"divp_UNet_pred.png")
plt.close()

print(f"Pred: Real part (min,max,mean-abs-value): \t{np.min(np.real(divP_FNO1))}\t{np.max(np.real(divP_FNO1))}\t{np.mean(np.abs(np.real(divP_FNO1)))}")
imdata = np.squeeze(np.real(divP_FNO1))
vmin=np.min(imdata)
vmax=np.max(imdata)
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
plt.figure()
plt.set_cmap("seismic")
plt.imshow(imdata,norm=norm)
plt.colorbar()
plt.savefig(f"divp_FNO_pred.png")
plt.close()