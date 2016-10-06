require 'cutorch'
require 'hzproc'
require 'image'
require 'math'
require 'paths'


cutorch.setDevice(2)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')



local folderName    = '../brainMRI/ABIDE/'
local targetFolder  = '../brainMRI/ABIDE/data/'
local imgDIR        = paths.dir(folderName)
table.sort(imgDIR)

local targetImgSize = 256
local total_iter    = 24000
local img_cnt = 1
local ref_cnt = 1
print(imgDIR)
local ref       = image.load(folderName .. imgDIR[ref_cnt + 2])

paths.mkdir(targetFolder)

function hzproc_Affine(img, trans_x, trans_y, scale_x, scale_y, shear_x, shear_y, rotation)
	-- affine transformation matrix
	mat = hzproc.Affine.Shift(trans_x, trans_y)
	mat = mat * hzproc.Affine.Scale(scale_x, scale_y)
	mat = mat * hzproc.Affine.Rotate(rotation)
	mat = mat * hzproc.Affine.Shear(shear_x, shear_y)
	-- affine mapping
	outs = hzproc.Transform.Fast(img:cuda(), mat);
	-- display the images
	-- image.display(O)
    return outs
end

print('generating data ......')
for iter = 1, total_iter do 

    if (iter%1000) == 0 and iter < 24000 then
        ref_cnt = ref_cnt + 1
        ref     = image.load(folderName .. imgDIR[ref_cnt + 2])
    end

    if (iter%1000) == 1 then
    
        local tmp_ref = image.scale(ref:float(), targetImgSize, targetImgSize)
        image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_1.png', tmp_ref)

        local trans_x = torch.uniform(-20, 20)
        local trans_y = torch.uniform(-10, 10)
        local scale_x = torch.uniform(0.8, 1.2)
        local scale_y = torch.uniform(0.8, 1.2)
        local shear_x = torch.uniform(-0.2,0.2)
        local shear_y = torch.uniform(-0.2,0.2)
        local rotation= torch.uniform(-20, 20)*math.pi/180

        local tform = torch.zeros(2,3)
        tform[1][1]   = scale_x*math.cos(rotation)
        tform[1][2]   = -shear_x*math.sin(rotation)
        tform[1][3]   = trans_x
        tform[2][1]   = scale_y*math.sin(rotation)
        tform[2][2]   = shear_y*math.cos(rotation)
        tform[2][3]   = trans_y

        local tmp_tar = hzproc_Affine(tmp_ref, trans_x, trans_y, scale_x, scale_y, shear_x, shear_y, rotation)
        local tar     = image.scale(tmp_tar:float(), targetImgSize, targetImgSize)
        image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_2.png', tar)
        torch.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_gt.t7', tform)
    
        print('apply trasformations to image# ' .. tostring(ref_cnt) .. '...')
    else

        local trans_x = torch.uniform(-10, 10)
        local trans_y = torch.uniform(-10, 10)
        local scale_x = torch.uniform(0.9, 1.1)
        local scale_y = torch.uniform(0.9, 1.1)
        local shear_x = torch.uniform(-0.1,0.1)
        local shear_y = torch.uniform(-0.1,0.1)
        local rotation= torch.uniform(-10, 10)*math.pi/180

        local tmp_ref = hzproc_Affine(ref, trans_x, trans_y, scale_x, scale_y, shear_x, shear_y, rotation)
        local tmp_ref2= image.scale(tmp_ref:float(), targetImgSize, targetImgSize)
        image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_1.png', tmp_ref2)

        -- ====================================
        -- ====================================

         trans_x = torch.uniform(-10, 10)
         trans_y = torch.uniform(-10, 10)
         scale_x = torch.uniform(0.9, 1.1)
         scale_y = torch.uniform(0.9, 1.1)
         shear_x = torch.uniform(-0.1,0.1)
         shear_y = torch.uniform(-0.1,0.1)
         rotation= torch.uniform(-10, 10)*math.pi/180

        local tform = torch.zeros(2,3)
        tform[1][1]   = scale_x*math.cos(rotation)
        tform[1][2]   = -shear_x*math.sin(rotation)
        tform[1][3]   = trans_x
        tform[2][1]   = scale_y*math.sin(rotation)
        tform[2][2]   = shear_y*math.cos(rotation)
        tform[2][3]   = trans_y

        local tmp_tar = hzproc_Affine(tmp_ref2, trans_x, trans_y, scale_x, scale_y, shear_x, shear_y, rotation)
        local tar     = image.scale(tmp_tar:float(), targetImgSize, targetImgSize)
        image.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_2.png', tar)
        torch.save(targetFolder .. 'brain_' .. string.format('%05d', img_cnt) .. '_gt.t7', tform)

    end

    img_cnt = img_cnt + 1
end

print('generating data DONE !!!! ')
print('========================================================')
