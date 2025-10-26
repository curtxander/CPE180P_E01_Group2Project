# datasets/dataset_utils.py
# Fixed for your specific Diamond Images Dataset structure

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiamondDatasetPreparator:
    """
    Prepare and organize diamond dataset for training
    
    Your Dataset Structure:
    raw_diamond_images/
    ├── web_scraped/
    │   ├── cushion/
    │   ├── emerald/
    │   ├── heart/
    │   ├── marquise/
    │   ├── oval/
    │   ├── pear/
    │   ├── princess/
    │   └── round/
    └── diamond_data.csv
    """
    
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Define categories based on your CSV data
        self.cut_categories = ['EX', 'VG', 'GD', 'FG']  # Based on your CSV
        self.shape_categories = ['cushion', 'emerald', 'heart', 'marquise', 'oval', 
                                'pear', 'princess', 'round']
        self.color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        
        # Detect dataset structure
        self.dataset_structure = self.detect_structure()
    
    def detect_structure(self):
        """Detect the structure of the downloaded dataset"""
        logger.info("Detecting dataset structure...")
        
        # Check for CSV file
        csv_files = list(self.source_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"Found CSV file: {csv_files[0]}")
            return "csv_based"
        
        # Check for web_scraped folder with shape subdirectories
        web_scraped_dir = self.source_dir / "web_scraped"
        if web_scraped_dir.exists():
            logger.info("Found web_scraped folder structure")
            return "web_scraped"
        
        logger.error("Could not detect dataset structure!")
        return "unknown"
    
    def create_directory_structure(self):
        """Create organized directory structure for dataset"""
        logger.info("Creating directory structure...")
        
        for split in ['train', 'val', 'test']:
            for category_type in ['cut', 'shape', 'color']:
                if category_type == 'cut':
                    categories = self.cut_categories
                elif category_type == 'shape':
                    categories = self.shape_categories
                else:
                    categories = self.color_categories
                
                for category in categories:
                    dir_path = self.target_dir / split / category_type / category
                    dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ Directory structure created successfully")
    
    def organize_from_csv(self, csv_file):
        """Organize dataset using the CSV file and web_scraped folder"""
        logger.info(f"Reading CSV file: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"CSV columns: {df.columns.tolist()}")
            logger.info(f"Total rows: {len(df)}")
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            # Find the correct columns based on your CSV structure
            image_col = None
            for col in df.columns:
                if 'path' in col.lower() or 'img' in col.lower():
                    image_col = col
                    break
            
            if not image_col:
                # Try to find by position (first column)
                image_col = df.columns[0]
                logger.info(f"Using first column as image path: {image_col}")
            
            # Find label columns
            shape_col = None
            cut_col = None
            color_col = None
            
            for col in df.columns:
                if 'shape' in col.lower():
                    shape_col = col
                elif 'cut' in col.lower():
                    cut_col = col
                elif 'color' in col.lower() or 'colour' in col.lower():
                    color_col = col
            
            logger.info(f"Detected columns - Image: {image_col}, Cut: {cut_col}, Shape: {shape_col}, Color: {color_col}")
            
            # Create temp directory for organized images
            temp_dir = self.target_dir / 'temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each row
            organized = {'cut': 0, 'shape': 0, 'color': 0}
            processed_images = set()
            
            for idx, row in df.iterrows():
                if pd.isna(row[image_col]):
                    continue
                    
                img_path_str = str(row[image_col]).strip()
                
                # Construct full image path
                if img_path_str.startswith('web_scraped/'):
                    img_path = self.source_dir / img_path_str
                else:
                    img_path = self.source_dir / "web_scraped" / img_path_str
                
                if not img_path.exists():
                    # Try alternative paths
                    alt_path = self.source_dir / img_path_str
                    if alt_path.exists():
                        img_path = alt_path
                    else:
                        logger.warning(f"Image not found: {img_path}")
                        continue
                
                # Skip if we've already processed this image
                if str(img_path) in processed_images:
                    continue
                processed_images.add(str(img_path))
                
                # Organize by shape
                if shape_col and pd.notna(row[shape_col]):
                    shape_value = str(row[shape_col]).strip().lower()
                    if shape_value in [s.lower() for s in self.shape_categories]:
                        # Find the correct case
                        actual_shape = next((s for s in self.shape_categories if s.lower() == shape_value.lower()), shape_value)
                        target_dir = temp_dir / 'shape' / actual_shape
                        target_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(img_path, target_dir / img_path.name)
                            organized['shape'] += 1
                        except Exception as e:
                            logger.warning(f"Error copying image {img_path}: {e}")
                
                # Organize by cut
                if cut_col and pd.notna(row[cut_col]):
                    cut_value = str(row[cut_col]).strip().upper()
                    # Map cut values to standard categories
                    cut_mapping = {
                        'EX': 'EX', 'EXCELLENT': 'EX',
                        'VG': 'VG', 'VERY GOOD': 'VG', 
                        'GD': 'GD', 'GOOD': 'GD',
                        'FG': 'FG', 'FINE GOOD': 'FG'
                    }
                    standardized_cut = cut_mapping.get(cut_value, cut_value)
                    if standardized_cut in self.cut_categories:
                        target_dir = temp_dir / 'cut' / standardized_cut
                        target_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(img_path, target_dir / img_path.name)
                            organized['cut'] += 1
                        except Exception as e:
                            logger.warning(f"Error copying image {img_path}: {e}")
                
                # Organize by color
                if color_col and pd.notna(row[color_col]):
                    color_value = str(row[color_col]).strip().upper()
                    if color_value in self.color_categories:
                        target_dir = temp_dir / 'color' / color_value
                        target_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(img_path, target_dir / img_path.name)
                            organized['color'] += 1
                        except Exception as e:
                            logger.warning(f"Error copying image {img_path}: {e}")
            
            logger.info(f"✓ Organized images - Cut: {organized['cut']}, Shape: {organized['shape']}, Color: {organized['color']}")
            
            # Update source directory to temp folder for splitting
            self.source_dir = temp_dir
            return True
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            import traceback
            traceback.print_exc()
            return False

    def organize_from_web_scraped(self):
        """Organize dataset directly from web_scraped folder structure"""
        logger.info("Organizing from web_scraped folder structure...")
        
        web_scraped_dir = self.source_dir / "web_scraped"
        if not web_scraped_dir.exists():
            logger.error("web_scraped directory not found!")
            return False
        
        # Create temp directory
        temp_dir = self.target_dir / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        organized = {'shape': 0}
        
        # Copy shape folders directly
        for shape in self.shape_categories:
            shape_dir = web_scraped_dir / shape
            if shape_dir.exists():
                target_shape_dir = temp_dir / 'shape' / shape
                target_shape_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all images
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_path in shape_dir.glob(ext):
                        try:
                            shutil.copy2(img_path, target_shape_dir / img_path.name)
                            organized['shape'] += 1
                        except Exception as e:
                            logger.warning(f"Error copying {img_path}: {e}")
        
        logger.info(f"✓ Organized {organized['shape']} shape images")
        
        # Update source directory to temp folder
        self.source_dir = temp_dir
        return True
    
    def validate_and_clean_images(self, min_size=(100, 100)):
        """Validate images and remove corrupted ones"""
        logger.info("Validating images...")
        
        valid_count = 0
        invalid_count = 0
        invalid_files = []
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(self.source_dir.rglob(ext))
        
        logger.info(f"Found {len(all_images)} image files to validate")
        
        for img_path in all_images:
            try:
                img = Image.open(img_path)
                img.verify()  # Verify image integrity
                
                # Reopen for size check
                img = Image.open(img_path)
                
                # Check minimum size
                if img.size[0] >= min_size[0] and img.size[1] >= min_size[1]:
                    valid_count += 1
                else:
                    logger.warning(f"Image too small: {img_path} - {img.size}")
                    invalid_count += 1
                    invalid_files.append(str(img_path))
                    
            except Exception as e:
                logger.warning(f"Invalid image {img_path}: {e}")
                invalid_count += 1
                invalid_files.append(str(img_path))
        
        logger.info(f"✓ Validation complete: {valid_count} valid, {invalid_count} invalid")
        
        # Save invalid files list
        if invalid_files:
            invalid_log = self.target_dir / 'invalid_images.txt'
            invalid_log.parent.mkdir(parents=True, exist_ok=True)
            with open(invalid_log, 'w') as f:
                f.write('\n'.join(invalid_files))
            logger.info(f"Invalid images list saved to: {invalid_log}")
        
        return valid_count, invalid_count
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train, validation, and test sets"""
        logger.info("Splitting dataset...")
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"
        
        stats = []
        total_images = 0
        
        # Process each category type
        for category_type in ['cut', 'shape', 'color']:
            if category_type == 'cut':
                categories = self.cut_categories
            elif category_type == 'shape':
                categories = self.shape_categories
            else:
                categories = self.color_categories
            
            logger.info(f"\nProcessing {category_type} categories...")
            
            for category in categories:
                # Find all images for this category
                category_dir = self.source_dir / category_type / category
                if not category_dir.exists():
                    logger.warning(f"Directory not found: {category_dir}")
                    continue
                
                # Get all image files
                images = list(category_dir.glob("*.jpg")) + \
                        list(category_dir.glob("*.jpeg")) + \
                        list(category_dir.glob("*.png")) + \
                        list(category_dir.glob("*.JPG")) + \
                        list(category_dir.glob("*.JPEG")) + \
                        list(category_dir.glob("*.PNG"))
                
                if len(images) == 0:
                    logger.warning(f"No images found in {category_dir}")
                    continue
                
                # Remove duplicates
                images = list(set(images))
                total_images += len(images)
                
                # Split images
                if len(images) < 3:
                    logger.warning(f"Not enough images in {category_dir} (found {len(images)})")
                    # Put all in train if very few images
                    train_imgs = images
                    val_imgs = []
                    test_imgs = []
                else:
                    train_imgs, temp_imgs = train_test_split(
                        images, test_size=(1 - train_ratio), random_state=42
                    )
                    
                    if len(temp_imgs) < 2:
                        val_imgs = []
                        test_imgs = temp_imgs
                    else:
                        val_imgs, test_imgs = train_test_split(
                            temp_imgs, 
                            test_size=test_ratio/(val_ratio + test_ratio),
                            random_state=42
                        )
                
                # Copy images to respective directories
                for split_name, img_list in [('train', train_imgs), 
                                             ('val', val_imgs), 
                                             ('test', test_imgs)]:
                    target_split_dir = self.target_dir / split_name / category_type / category
                    
                    for img_path in img_list:
                        target_path = target_split_dir / img_path.name
                        try:
                            shutil.copy2(img_path, target_path)
                        except Exception as e:
                            logger.warning(f"Error copying {img_path}: {e}")
                
                # Log statistics
                logger.info(f"  {category}: train={len(train_imgs)}, "
                          f"val={len(val_imgs)}, test={len(test_imgs)}")
                
                stats.append({
                    'category_type': category_type,
                    'category': category,
                    'train': len(train_imgs),
                    'val': len(val_imgs),
                    'test': len(test_imgs),
                    'total': len(images)
                })
        
        logger.info(f"✓ Dataset split complete - Total images processed: {total_images}")
        return stats
    
    def generate_statistics(self):
        """Generate dataset statistics"""
        logger.info("\nGenerating dataset statistics...")
        
        stats = {
            'split': [],
            'category_type': [],
            'category': [],
            'count': []
        }
        
        for split in ['train', 'val', 'test']:
            for category_type in ['cut', 'shape', 'color']:
                if category_type == 'cut':
                    categories = self.cut_categories
                elif category_type == 'shape':
                    categories = self.shape_categories
                else:
                    categories = self.color_categories
                
                for category in categories:
                    dir_path = self.target_dir / split / category_type / category
                    if dir_path.exists():
                        count = len(list(dir_path.glob("*.jpg"))) + \
                               len(list(dir_path.glob("*.jpeg"))) + \
                               len(list(dir_path.glob("*.png"))) + \
                               len(list(dir_path.glob("*.JPG"))) + \
                               len(list(dir_path.glob("*.JPEG"))) + \
                               len(list(dir_path.glob("*.PNG")))
                        
                        stats['split'].append(split)
                        stats['category_type'].append(category_type)
                        stats['category'].append(category)
                        stats['count'].append(count)
        
        df = pd.DataFrame(stats)
        
        # Save statistics
        stats_path = self.target_dir / 'dataset_statistics.csv'
        df.to_csv(stats_path, index=False)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS SUMMARY")
        logger.info("="*60)
        
        for split in ['train', 'val', 'test']:
            split_data = df[df['split'] == split]
            total = split_data['count'].sum()
            logger.info(f"\n{split.upper()}: {total} images")
            
            for cat_type in ['cut', 'shape', 'color']:
                type_data = split_data[split_data['category_type'] == cat_type]
                type_total = type_data['count'].sum()
                if type_total > 0:
                    logger.info(f"  {cat_type}: {type_total} images")
        
        logger.info(f"\n✓ Statistics saved to: {stats_path}")
        logger.info("="*60)
        
        return df
    
    def create_metadata_file(self):
        """Create metadata file with dataset information"""
        metadata = {
            'dataset_name': 'Diamond Images Dataset',
            'source': 'Web Scraped Diamond Images',
            'categories': {
                'cut': self.cut_categories,
                'shape': self.shape_categories,
                'color': self.color_categories
            },
            'splits': ['train', 'val', 'test'],
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            'image_format': ['jpg', 'jpeg', 'png'],
            'target_size': [224, 224],
            'preprocessing': [
                'resize',
                'normalize',
                'denoise',
                'contrast_enhancement'
            ]
        }
        
        metadata_path = self.target_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to: {metadata_path}")

def preprocess_kaggle_dataset(kaggle_dataset_path, output_path):
    """
    Preprocess the Diamond Images Dataset
    
    Args:
        kaggle_dataset_path: Path to downloaded dataset
        output_path: Path where processed dataset will be saved
    """
    logger.info("="*60)
    logger.info("DIAMOND DATASET PREPROCESSING")
    logger.info("="*60)
    
    preparator = DiamondDatasetPreparator(kaggle_dataset_path, output_path)
    
    # Step 0: Handle different dataset structures
    if preparator.dataset_structure == "csv_based":
        csv_files = list(preparator.source_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"Processing CSV-based dataset: {csv_files[0]}")
            success = preparator.organize_from_csv(csv_files[0])
            if not success:
                logger.error("Failed to organize dataset from CSV, trying web_scraped structure...")
                success = preparator.organize_from_web_scraped()
    elif preparator.dataset_structure == "web_scraped":
        success = preparator.organize_from_web_scraped()
    else:
        logger.error("Unknown dataset structure!")
        return None
    
    if not success:
        logger.error("Failed to organize dataset!")
        return None
    
    # Step 1: Create directory structure
    preparator.create_directory_structure()
    
    # Step 2: Validate images
    valid, invalid = preparator.validate_and_clean_images()
    
    if valid == 0:
        logger.error("No valid images found! Please check dataset structure.")
        logger.info("Expected structure:")
        logger.info("  raw_diamond_images/")
        logger.info("    ├── web_scraped/")
        logger.info("    │   ├── cushion/")
        logger.info("    │   ├── emerald/")
        logger.info("    │   └── ...")
        logger.info("    └── diamond_data.csv")
        return None
    
    # Step 3: Split dataset
    stats = preparator.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    if not stats:
        logger.error("Dataset split failed - no images processed")
        return None
    
    # Step 4: Generate statistics
    df = preparator.generate_statistics()
    
    # Step 5: Create metadata
    preparator.create_metadata_file()
    
    # Clean up temp directory if it exists
    temp_dir = Path(output_path) / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logger.info("✓ Cleaned up temporary files")
    
    logger.info("\n" + "="*60)
    logger.info("DATASET PREPROCESSING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Total valid images: {valid}")
    logger.info(f"Total invalid images: {invalid}")
    logger.info(f"Output directory: {output_path}")
    logger.info("="*60)
    
    return df

if __name__ == "__main__":
    # Configuration
    KAGGLE_DATASET = "./raw_diamond_images"
    OUTPUT_DATASET = "./diamond_dataset"
    
    # Check if source directory exists
    if not Path(KAGGLE_DATASET).exists():
        logger.error(f"Source directory not found: {KAGGLE_DATASET}")
        logger.error("\nPlease ensure the dataset is extracted to: ./datasets/raw_diamond_images/")
        exit(1)
    
    # Run preprocessing
    try:
        stats = preprocess_kaggle_dataset(KAGGLE_DATASET, OUTPUT_DATASET)
        
        if stats is not None:
            logger.info("\n✓ Dataset ready for training!")
            total_images = stats['count'].sum() if hasattr(stats, 'count') else "Unknown"
            logger.info(f"Total images: {total_images}")
            logger.info(f"\nNext step: Train models")
            logger.info("  python train_models.py --data_dir ./diamond_dataset")
        else:
            logger.error("\n✗ Dataset preprocessing failed!")
            logger.error("Please check the error messages above and fix the issues.")
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()