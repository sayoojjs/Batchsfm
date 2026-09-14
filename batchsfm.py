#BatchSFM v1.1 by 9sAE7

import subprocess
import shutil
from pathlib  import Path

TRUENAS = Path("/mnt/truenas/scanned_dataset")
INPUT = TRUENAS / "input"
OUTPUT = TRUENAS / "ml"
WORK = Path.home() / "scan_work"

YELLOW = '\033[33m'
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'


def run(cmd):
        print("\n$", " ".join(map(str, cmd)))
        subprocess.run(cmd, check=True)

print("------------------------------------------------------------------")
print(r"""
   _ ,         
  _-_ _,,           ,       ,,            -_-/    ,- -          
     -/  )    _    ||       ||           (_ /    _||_           
    ~||_<    < \, =||=  _-_ ||/\\       (_ --_  ' ||   \\/\\/\\ 
     || \\   /-||  ||  ||   || ||         --_ )   ||   || || || 
     ,/--|| (( ||  ||  ||   || ||        _/  ))   |,   || || || 
    _--_-'   \/\\  \\, \\,/ \\ |/       (_-_-   _-/    \\ \\ \\ 
   (                          _/
""")
print("------------------------------------------------------------------")

FPS = int(input(GREEN + "Enter how many images sequence needed (Counted in FPS):" +RESET))

IMG_WIDTH = int(input(YELLOW + "Enter the Image width:" + RESET))
IMG_HEIGHT = int(input(YELLOW + "Enter the Image height:" + RESET))

MAX_FEATURES = int(input(GREEN + "Enter the number of SIFT features need per image:"))
PATCH_MATCH = input("Enable patch match stereo? (y/n):").lower() == "y"

TRAINASK = input("Do you want to train the dataset? (y/n):" + RESET).lower() == "y" 
if TRAINASK:
        GS_ITERATIONS = int(input(YELLOW + "How many iterations (5000/15000/30000):"))
        GS_RESOLUTION = int(input("Choose the downscale factor (1/2/../8/9/):")) 
        GS_DENSEITR = int(input("How many iteration you need until densification stops(5000/15000/30000):"))  
        GS_SH = input("Enable Spherical Harmonics ? (y/n):" + RESET).strip().lower() == "y"   
else:
        print(YELLOW + "Training skipped!" + RESET)

MAX_IMG_SIZE = max(IMG_WIDTH, IMG_HEIGHT)


if GS_SH:
        GS_SH_SET = 1
else:
        GS_SH_SET = 0

#Video fetching loop
videos = list(INPUT.glob("*.MOV")) + list(INPUT.glob("*.mov"))

processed_datasets = []

for video in videos:

        name = video.stem
        dataset = WORK / name
        gsplats = dataset / "dataset.gsplats"
        images = gsplats / "images"
        sparse = gsplats / "sparse"
        dense = gsplats / "dense" 

        shutil.rmtree(dataset, ignore_errors=True)
        images.mkdir(parents=True)
        sparse.mkdir()
        dense.mkdir()

        shutil.copy2(video, gsplats / video.name)

        print (YELLOW + f"\n\n                     Processing: {name}\n\n" + RESET)

        if not video:
                print(RED + "                      Videos not found!" + RESET)

        #image making
        print ("--------------Frame Making-------------------")
     
        run([
                "ffmpeg",
                "-i", gsplats / video.name,
                "-vf", f"fps={FPS}, scale={IMG_WIDTH}:{IMG_HEIGHT}",
                "-q:v", "2",
                images / "frame_%05d.jpg"
        ])

        print ("------------Feature Extraction---------------")

        run([

                "colmap", "feature_extractor",
                "--database_path", str(gsplats / "database.db"),
                "--image_path", str(images),
                "--ImageReader.single_camera", "1",
                "--FeatureExtraction.use_gpu", "1",
                "--FeatureExtraction.gpu_index", "0",
                "--FeatureExtraction.max_image_size", str(MAX_IMG_SIZE),
                "--SiftExtraction.max_num_features", str(MAX_FEATURES),
                "--SiftExtraction.estimate_affine_shape", "1",
                "--SiftExtraction.domain_size_pooling", "1"
        ])


       #sequential matching
        print ("-------COLMAP Exaustive Matching -------")
       
        run([
                "colmap", "sequential_matcher",
                "--database_path", str(gsplats / "database.db"),
                "--SequentialMatching.overlap", "10",
                "--SequentialMatching.loop_detection", "1",
                "--FeatureMatching.use_gpu", "1",
                "--FeatureMatching.gpu_index", "0",
                "--FeatureMatching.guided_matching", "1"
               ])


        print ("----------Sparse Reconstruction-----------")

        run([

                "colmap", "mapper", 
                "--database_path", str(gsplats / "database.db"),
                "--image_path", str(images),
                "--output_path", str(sparse)

        ])


        if not any(sparse.glob("*/cameras.bin")):
                print(RED + "SPARSE RECONSTRUCTION FAILED" + RESET)
                continue




        print ("---------Prepare Dense Reconstruction--------")
        
        run([
        
                "colmap", "image_undistorter",
                "--image_path", str(images),
                "--input_path", str(sparse / "0"),
                "--output_path", str(dense),
                "--output_type", "COLMAP",
                "--max_image_size", str(MAX_IMG_SIZE)
        ])




        dense_sparse = dense / "sparse"
        dense_sparse_0 = dense_sparse / "0"
        dense_sparse_0.mkdir(parents=True, exist_ok=True)

        reconstruction_ok = True
        for filename in ["cameras.bin", "images.bin", "points3D.bin"]:
                source = dense_sparse / filename
                destination = dense_sparse_0 / filename

                if source.exists():

                        shutil.move(str(source), str(destination))

                        print(GREEN + f"Moved {filename} -> dense/sparse/0/" +  RESET)

                else:

                        print(
                        RED + f"Missing {filename} after image undistortion" + RESET)



        if PATCH_MATCH:
                print(YELLOW + "---------- Patch Match Stereo ----------" + RESET)
                run([
                        "colmap", "patch_match_stereo",
                        "--workspace_path", str(dense),
                        "--workspace_format", "COLMAP",
                        "--PatchMatchStereo.max_image_size", str(MAX_IMG_SIZE),
                        "--PatchMatchStereo.geom_consistency", "true"
                        ])

                print(YELLOW + "---------- Stereo Fusion ----------" + RESET)
                run([
                        "colmap", "stereo_fusion",
                        "--workspace_path", str(dense),
                        "--workspace_format", "COLMAP",
                        "--input_type", "geometric",
                        "--output_path", str(dense / "fused.ply")
                ])
                
        else:
                print(YELLOW + "Patch match stereo skipped." + RESET)



        if TRAINASK: 
                print(YELLOW +"\n---------- 3D Gaussian Splatting Training ----------" + RESET)
                run([
                        "/home/sayooj/miniconda3/envs/gaussian/bin/python",  #Replace this with your location
                        "/home/sayooj/gaussian-splatting/train.py",          #Replace this with your location
                        "--source_path", str(dense),
                        "--model_path", str(gsplats / "output"),  
                        "--iterations", str(GS_ITERATIONS),
                        "--resolution", str(GS_RESOLUTION),
                        "--densify_until_iter", str(GS_DENSEITR),
                        "--densify_grad_threshold", "0.0004",
                        "--sh_degree", str(GS_SH_SET),
                        "--test_iterations", "-1",
                        "--quiet"
                ])

        else:
                print(RED + "Training skipped by user." + RESET)

       
        print ("------- Copy dataset back to TrueNAS-----")
        
        output = OUTPUT / name
        output.mkdir(parents=True, exist_ok=True)

        run([
                "rsync", "-a",
                str(dataset) + "/",
                str(output) + "/"
        ])

        processed_datasets.append((dataset, images))

        print(GREEN + f"\nFinished Processing: {name}" +RESET)    


if processed_datasets:
        CLR_DATA = input(GREEN + "Do you want to clear local cache? (y/n):" + RESET).lower() == "y"

        if CLR_DATA:
           for dataset, images in processed_datasets:
                shutil.rmtree(dataset, ignore_errors=True)
                shutil.rmtree(images, ignore_errors=True)
                print(YELLOW + "Local cache removed from the disk" + RESET)
        else:
                print(YELLOW + "Clearing cache skipped" + RESET)
else:
   print(RED+ "No videos were successfully processed!" + RESET)
                        

