# datasets/dataset_utils.py
# Final version for actual Kaggle Diamond Images Dataset

import os
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiamondDatasetPreparator:
    """
    Handle actual Kaggle Diamond Images Dataset:
    - CSV with columns: path_to_img, shape, colour, cut
    - Cut values: EX, VG, GD, FG (abbreviated)
    - Color values: Extended range (D-Z, fancy colors)
    """
    
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Map abbreviated cut values to full names
        self.cut_mapping = {
            'EX': 'Excellent',
            'VG': 'Very Good', 
            'GD': 'Good',
            'FG': 'Fair',
            'ID': 'Ideal',
            'PR': 'Poor'
        }
        
        # Standard categories for training
        self.cut_categories = ['Ideal', 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
        self.shape_categories = ['Round', 'Princess', 'Emerald', 'Asscher', 'Oval', 
                                'Radiant', 'Cushion', 'Marquise', 'Pear', 'Heart']
        # Extended color range to match dataset
        self.color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        
        # Find and load CSV
        self.csv_file = self.find_csv_file()
        self.metadata_df = None
        
        if self.csv_file:
            logger.info(f"✓ Found CSV: {self.csv_file.name}")
            self.load_metadata()
        else:
            raise FileNotFoundError("No CSV file found!")
    
    def find_csv_file(self):
        """Find CSV file"""
        for csv_file in self.source_dir.glob("*.csv"):
            return csv_file
        for csv_file in self.source_dir.rglob("*.csv"):
            return csv_file
        return None
    
    def load_metadata(self):
        """Load CSV metadata"""
        try:
            df = pd.read_csv(self.csv_file)
            logger.info(f"✓ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            self.metadata_df = df
            
            # Show sample
            logger.info(f"\nSample data:")
            logger.info(df.head(3).to_string())
            
            # Show unique values
            if 'cut' in df.columns:
                logger.info(f"\nUnique cut values: {sorted(df['cut'].dropna().unique())}")
            if 'colour' in df.columns:
                logger.info(f"Unique colour values: {sorted(df['colour'].dropna().unique())}")
            if 'shape' in df.columns:
                logger.info(f"Unique shape values: {sorted(df['shape'].dropna().unique())}")
                
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def normalize_cut_value(self, cut_abbr):
        """Convert abbreviated cut to full name"""
        cut_abbr = str(cut_abbr).strip().upper()
        return self.cut_mapping.get(cut_abbr, None)
    
    def normalize_shape_value(self, shape):
        """Normalize shape value"""
        shape = str(shape).strip().lower()
        # Capitalize first letter
        shape_normalized = shape.capitalize()
        
        if shape_normalized in self.shape_categories:
            return shape_normalized
        return None
    
    def normalize_color_value(self, color):
        """Normalize color value"""
        color = str(color).strip().upper()
        
        # Handle fancy colors - skip them for now
        if 'FANCY' in color:
            return None
        
        # Handle color ranges (e.g., Y-Z, S-T)
        if '-' in color:
            # Use the first color in range
            color = color.split('-')[0]
        
        # Only use D-N for consistency
        if color in self.color_categories:
            return color
        
        return None
    
    def create_directory_structure(self):
        """Create output directories"""
        logger.info("\nCreating directory structure...")
        
        for split in ['train', 'val', 'test']:
            for cat_type in ['cut', 'shape', 'color']:
                categories = getattr(self, f'{cat_type}_categories')
                for category in categories:
                    dir_path = self.target_dir / split / cat_type / category
                    dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ Directories created")
    
    def organize_dataset(self):
        """Organize dataset from CSV"""
        logger.info("\nOrganizing dataset...")
        
        temp_dir = self.target_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        organized = {'cut': 0, 'shape': 0, 'color': 0}
        skipped = {'no_file': 0, 'invalid_cut': 0, 'invalid_shape': 0, 'invalid_color': 0}
        
        total_rows = len(self.metadata_df)
        
        for idx, row in self.metadata_df.iterrows():
            # Get image path from CSV
            img_path_str = str(row['path_to_img'])
            img_path = self.source_dir / img_path_str
            
            # Check if file exists
            if not img_path.exists():
                # Try without web_scraped prefix
                img_path = self.source_dir / Path(img_path_str).name
                if not img_path.exists():
                    # Try in shape folder
                    shape_from_path = Path(img_path_str).parent.name
                    img_path = self.source_dir / shape_from_path / Path(img_path_str).name
                    if not img_path.exists():
                        skipped['no_file'] += 1
                        continue
            
            img_name = img_path.name
            
            # Process shape
            if 'shape' in row and pd.notna(row['shape']):
                shape_norm = self.normalize_shape_value(row['shape'])
                if shape_norm:
                    shape_dir = temp_dir / 'shape' / shape_norm
                    shape_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, shape_dir / img_name)
                    organized['shape'] += 1
                else:
                    skipped['invalid_shape'] += 1
            
            # Process cut
            if 'cut' in row and pd.notna(row['cut']):
                cut_norm = self.normalize_cut_value(row['cut'])
                if cut_norm and cut_norm in self.cut_categories:
                    cut_dir = temp_dir / 'cut' / cut_norm
                    cut_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, cut_dir / img_name)
                    organized['cut'] += 1
                else:
                    skipped['invalid_cut'] += 1
            
            # Process color
            if 'colour' in row and pd.notna(row['colour']):
                color_norm = self.normalize_color_value(row['colour'])
                if color_norm:
                    color_dir = temp_dir / 'color' / color_norm
                    color_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, color_dir / img_name)
                    organized['color'] += 1
                else:
                    skipped['invalid_color'] += 1
            
            if (idx + 1) % 5000 == 0:
                logger.info(f"Progress: {idx + 1}/{total_rows} rows processed...")
        
        logger.info(f"\n✓ Organization complete:")
        logger.info(f"  Shape: {organized['shape']} images")
        logger.info(f"  Cut: {organized['cut']} images")
        logger.info(f"  Color: {organized['color']} images")
        
        logger.info(f"\nSkipped:")
        logger.info(f"  Files not found: {skipped['no_file']}")
        logger.info(f"  Invalid cut values: {skipped['invalid_cut']}")
        logger.info(f"  Invalid shape values: {skipped['invalid_shape']}")
        logger.info(f"  Invalid/fancy colors: {skipped['invalid_color']}")
        
        self.source_dir = temp_dir
        return True
    
    def validate_images(self, min_size=(50, 50)):
        """Validate images"""
        logger.info("\nValidating images...")
        
        valid = 0
        invalid = 0
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.source_dir.rglob(ext):
                try:
                    img = Image.open(img_path)
                    img.verify()
                    img = Image.open(img_path)
                    if img.size[0] >= min_size[0] and img.size[1] >= min_size[1]:
                        valid += 1
                    else:
                        invalid += 1
                except:
                    invalid += 1
        
        logger.info(f"✓ Valid: {valid}, Invalid: {invalid}")
        return valid, invalid
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split into train/val/test"""
        logger.info("\nSplitting dataset...")
        
        stats = []
        
        for cat_type in ['cut', 'shape', 'color']:
            categories = getattr(self, f'{cat_type}_categories')
            logger.info(f"\nProcessing {cat_type}...")
            
            for category in categories:
                cat_dir = self.source_dir / cat_type / category
                if not cat_dir.exists():
                    continue
                
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images.extend(list(cat_dir.glob(ext)))
                
                if len(images) == 0:
                    continue
                
                if len(images) < 3:
                    logger.warning(f"  {category}: only {len(images)} images")
                    continue
                
                # Split
                train_imgs, temp_imgs = train_test_split(images, test_size=(1-train_ratio), random_state=42)
                
                if len(temp_imgs) >= 2:
                    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
                else:
                    val_imgs = []
                    test_imgs = temp_imgs
                
                # Copy files
                for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
                    target_dir = self.target_dir / split_name / cat_type / category
                    for img_path in img_list:
                        shutil.copy2(img_path, target_dir / img_path.name)
                
                logger.info(f"  {category}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
                
                stats.append({
                    'category_type': cat_type,
                    'category': category,
                    'train': len(train_imgs),
                    'val': len(val_imgs),
                    'test': len(test_imgs),
                    'total': len(images)
                })
        
        return stats
    
    def generate_statistics(self):
        """Generate statistics"""
        logger.info("\nGenerating statistics...")
        
        stats = []
        for split in ['train', 'val', 'test']:
            for cat_type in ['cut', 'shape', 'color']:
                categories = getattr(self, f'{cat_type}_categories')
                for category in categories:
                    dir_path = self.target_dir / split / cat_type / category
                    if dir_path.exists():
                        count = len(list(dir_path.glob('*.[jJ]*[gG]'))) + len(list(dir_path.glob('*.[pP][nN][gG]')))
                        stats.append({
                            'split': split,
                            'category_type': cat_type,
                            'category': category,
                            'count': count
                        })
        
        df = pd.DataFrame(stats)
        df.to_csv(self.target_dir / 'dataset_statistics.csv', index=False)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("FINAL DATASET STATISTICS")
        logger.info("="*60)
        
        for split in ['train', 'val', 'test']:
            split_data = df[df['split'] == split]
            total = split_data['count'].sum()
            logger.info(f"\n{split.upper()}: {total} images")
            
            for cat_type in ['cut', 'shape', 'color']:
                type_data = split_data[split_data['category_type'] == cat_type]
                type_total = type_data['count'].sum()
                logger.info(f"  {cat_type}: {type_total} images")
        
        logger.info("="*60)
        
        return df

def preprocess_kaggle_dataset(source_dir, output_dir):
    """Main preprocessing"""
    logger.info("="*60)
    logger.info("AUTOGEMGRADE - DATASET PREPROCESSING")
    logger.info("="*60)
    
    prep = DiamondDatasetPreparator(source_dir, output_dir)
    
    prep.create_directory_structure()
    
    success = prep.organize_dataset()
    if not success:
        return None
    
    valid, invalid = prep.validate_images()
    
    if valid == 0:
        logger.error("No valid images!")
        return None
    
    stats = prep.split_dataset()
    
    if not stats:
        logger.error("Split failed!")
        return None
    
    df = prep.generate_statistics()
    
    # Cleanup
    temp_dir = Path(output_dir) / 'temp'
    if temp_dir.exists():
        logger.info("\nCleaning up...")
        shutil.rmtree(temp_dir)
    
    logger.info("\n" + "="*60)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Output: {output_dir}")
    logger.info("\nNext step:")
    logger.info("  cd ../cloud-backend")
    logger.info("  python train_models_pytorch.py --data_dir ../datasets/diamond_dataset")
    logger.info("="*60)
    
    return df

if __name__ == "__main__":
    SOURCE = "./raw_diamond_images"
    OUTPUT = "./diamond_dataset"
    
    if not Path(SOURCE).exists():
        logger.error(f"Source not found: {SOURCE}")
        exit(1)
    
    try:
        result = preprocess_kaggle_dataset(SOURCE, OUTPUT)
        if result is not None:
            logger.info("\n✅ Dataset ready for training!")
            logger.info(f"Total images: {result['count'].sum()}")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()