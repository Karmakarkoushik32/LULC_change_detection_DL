from inference import Inferencer
from utils import prepare_image_pair, write_data, polygonize
from config import CLASS_MAP, TARGET_RES
from config import IMAGE_PATCH_SIZE
from change_metrics import compute_change_metrics

import os
import json
import uuid
import rasterio
import logging
import traceback

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# PATHS
parent = os.path.dirname(__file__)
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(parent, "final_model/resunet_scripted.pt")
print(model_path)
# LOAD MODEL
logger.info("Loading model...")
inferencer = Inferencer(model_path=model_path, patch_size=IMAGE_PATCH_SIZE)
logger.info("Model loaded successfully.")


# MAIN PIPELINE
def main(image_path_1: str, image_path_2: str):

    try:
        logger.info("Starting pipeline...")

        # VALIDATE INPUTS
        logger.info("Validating input image paths...")

        if (not os.path.exists(image_path_1)) or (not os.path.exists(image_path_2)):
            raise RuntimeError("image_path_1 or image_path_2 or both are invalid!")

        logger.info("Input validation successful.")

        # CREATE RUN DIRECTORY
        run_id = str(uuid.uuid4())
        run_dir = os.path.join(output_dir, f"RUN_ID={run_id}")
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Run directory created: {run_dir}")

        # RECTIFICATION
        logger.info("Preparing / rectifying image pair...")

        data1, data2, transform1, transform2, target_crs = prepare_image_pair(
            tif1_path=image_path_1, tif2_path=image_path_2
        )

        logger.info("Image pair rectification completed.")

        # SAVE RECTIFIED IMAGES
        rectified_image_path_1 = os.path.join(run_dir, "processed_1.tif")
        rectified_image_path_2 = os.path.join(run_dir, "processed_2.tif")

        logger.info("Writing rectified images...")
        write_data(rectified_image_path_1, data1, transform1, target_crs)
        write_data(rectified_image_path_2, data2, transform2, target_crs)

        logger.info("Rectified images written successfully.")

        # INFERENCE

        segmented_image_path_1 = os.path.join(run_dir, "segmented_1.tif")
        segmented_image_path_2 = os.path.join(run_dir, "segmented_2.tif")

        logger.info("Running inference on image 1...")
        inferencer.predict(rectified_image_path_1, segmented_image_path_1)
        logger.info("Inference completed for image 1.")

        logger.info("Running inference on image 2...")
        inferencer.predict(rectified_image_path_2, segmented_image_path_2)
        logger.info("Inference completed for image 2.")

        # POLYGONIZATION

        segmented_polygonized_image_path_1 = os.path.join(
            run_dir, "segmented_polygonized_1.geojson"
        )

        segmented_polygonized_image_path_2 = os.path.join(
            run_dir, "segmented_polygonized_2.geojson"
        )

        logger.info("Opening segmentation rasters...")

        with (
            rasterio.open(segmented_image_path_1, masked=True) as src_1,
            rasterio.open(segmented_image_path_2, masked=True) as src_2,
        ):

            segmented_1 = src_1.read(1)
            segmented_2 = src_2.read(1)

            logger.info("Polygonizing segmentation 1...")

            feature_df1 = polygonize(
                segmented=segmented_1,
                class_mappings=CLASS_MAP,
                transform=src_1.transform,
                crs=src_1.crs,
                skip_classes=[0],
            )

            logger.info(f"Segmentation 1 polygons: {len(feature_df1)}")
            logger.info("Polygonizing segmentation 2...")

            feature_df2 = polygonize(
                segmented=segmented_2,
                class_mappings=CLASS_MAP,
                transform=src_2.transform,
                crs=src_2.crs,
                skip_classes=[0],
            )

            logger.info(f"Segmentation 2 polygons: {len(feature_df2)}")
            logger.info("Writing polygonized outputs...")
            feature_df1.to_file(segmented_polygonized_image_path_1)
            feature_df2.to_file(segmented_polygonized_image_path_2)
            logger.info("Polygonized outputs written.")

            # CHANGE METRICS

            logger.info("Computing change metrics...")

            change_metrics_path = os.path.join(run_dir, "change_metrics.json")

            compute_change_metrics(
                segmented_1=segmented_1,
                segmented_2=segmented_2,
                class_mappings=CLASS_MAP,
                pixel_size_m=TARGET_RES,
                output_json_path=change_metrics_path,
            )

            logger.info(f"Change metrics written: {change_metrics_path}")

        logger.info("Pipeline completed successfully.")

        # update runs.json
        runs_maifest_path = os.path.join(output_dir, "runs.json")
        if os.path.exists(runs_maifest_path):
            with open(runs_maifest_path, "r") as fp:
                runs = json.load(fp)
        else:
            runs = []

        with open(runs_maifest_path, "w") as fwp:
            runs.append({"id": run_id, "path": f"RUN_ID={run_id}"})
            json.dump(runs, fwp)


        return {
            "run_dir": run_dir,
            "segmented_1": segmented_image_path_1,
            "segmented_2": segmented_image_path_2,
            "polygonized_1": segmented_polygonized_image_path_1,
            "polygonized_2": segmented_polygonized_image_path_2,
            "change_metrics": change_metrics_path,
        }

    except Exception as e:
        logger.error("Pipeline failed!")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise
