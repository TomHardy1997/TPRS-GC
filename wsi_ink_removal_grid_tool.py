"""
WSI Grid Selector - Interactive Polygon-based Grid Generation Tool

This tool allows users to interactively select regions on Whole Slide Images (WSI)
by drawing polygons and automatically generates a grid of coordinates within the
selected regions for further analysis.

Author: JianXin Ji
"""

import os
import cv2
import h5py
import numpy as np
from PIL import Image
from histolab.slide import Slide
import openslide


class WSIGridSelector:
    """
    Interactive WSI grid selector for pathology image analysis.
    
    This class provides functionality to:
    1. Load and scale WSI files
    2. Interactive polygon drawing on thumbnail images
    3. Generate grid coordinates within selected regions
    4. Save coordinates for downstream analysis
    """
    
    def __init__(self, svs_file_path, output_dir='.', scale_factor=100):
        """
        Initialize the WSI Grid Selector.
        
        Args:
            svs_file_path (str): Path to the SVS file
            output_dir (str): Directory for output files
            scale_factor (int): Scale factor for thumbnail generation
        """
        self.svs_file_path = svs_file_path
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.slide = None
        self.thumbnail_image = None
        self.clone = None
        self.points = []
        self.grid_coordinates = []
        self.rect_size = 512  # Default grid size
        
    def load_slide(self):
        """Load the WSI slide and determine appropriate grid size."""
        try:
            self.slide = Slide(self.svs_file_path, self.output_dir)
            
            # Determine grid size based on objective power
            objective_power = self._get_objective_power()
            if objective_power and float(objective_power) == 40:
                self.rect_size = 1024
            elif objective_power and float(objective_power) == 20:
                self.rect_size = 512
            else:
                self.rect_size = 512  # Default value
                
            print(f"Slide loaded successfully. Grid size: {self.rect_size}px")
            return True
            
        except Exception as e:
            print(f"Error loading slide: {e}")
            return False
    
    def _get_objective_power(self):
        """
        Extract objective power from slide metadata.
        
        Returns:
            float or None: Objective power value
        """
        try:
            # Try to get objective power from slide properties
            # This is a placeholder - actual implementation depends on slide format
            return 40  # Default assumption
        except:
            return None
    
    def generate_thumbnail(self, output_filename='thumbnail.png'):
        """
        Generate and save a scaled thumbnail of the WSI.
        
        Args:
            output_filename (str): Name of the output thumbnail file
            
        Returns:
            bool: Success status
        """
        try:
            scaled_image = self.slide.scaled_image(scale_factor=self.scale_factor)
            thumbnail_path = os.path.join(self.output_dir, output_filename)
            scaled_image.save(thumbnail_path)
            
            # Load thumbnail for OpenCV processing
            self.thumbnail_image = cv2.imread(thumbnail_path)
            self.clone = self.thumbnail_image.copy()
            
            print(f"Thumbnail generated and saved as {output_filename}")
            print(f"Image scaled down by factor of {self.scale_factor}")
            return True
            
        except Exception as e:
            print(f"Error generating thumbnail: {e}")
            return False
    
    def _mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for interactive polygon drawing.
        
        Args:
            event: OpenCV mouse event
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click - add point
            self.points.append((x, y))
            # Draw current point on thumbnail
            cv2.circle(self.clone, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("WSI Grid Selector", self.clone)
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click - complete polygon
            if len(self.points) > 2:  # Need at least 3 points for polygon
                self._fill_rectangles_in_polygon(self.points)
                cv2.imshow("WSI Grid Selector", self.clone)
            
            # Reset for next polygon
            self.clone = self.thumbnail_image.copy()
            self.points = []
    
    def _fill_rectangles_in_polygon(self, polygon_points):
        """
        Fill rectangles within the drawn polygon and record coordinates.
        
        Args:
            polygon_points (list): List of polygon vertices in thumbnail coordinates
        """
        # Convert thumbnail coordinates to original image coordinates
        polygon_points_original = [
            (int(x * self.scale_factor), int(y * self.scale_factor)) 
            for x, y in polygon_points
        ]
        
        # Create mask for polygon region
        original_height = self.thumbnail_image.shape[0] * self.scale_factor
        original_width = self.thumbnail_image.shape[1] * self.scale_factor
        mask = np.zeros((int(original_height), int(original_width)), dtype=np.uint8)
        
        # Fill polygon on mask
        cv2.fillPoly(mask, [np.array(polygon_points_original)], 255)
        
        # Generate grid within polygon
        for y in range(0, int(original_height) - self.rect_size, self.rect_size):
            for x in range(0, int(original_width) - self.rect_size, self.rect_size):
                # Check if rectangle center is within polygon
                center_y, center_x = y + self.rect_size // 2, x + self.rect_size // 2
                
                if (center_y < mask.shape[0] and center_x < mask.shape[1] and 
                    mask[center_y, center_x] == 255):
                    
                    # Record coordinates in original resolution
                    self.grid_coordinates.append(
                        ((x, y), (x + self.rect_size, y + self.rect_size))
                    )
                    
                    # Draw rectangle on thumbnail for visualization
                    scaled_x = int(x / self.scale_factor)
                    scaled_y = int(y / self.scale_factor)
                    scaled_size = int(self.rect_size / self.scale_factor)
                    
                    cv2.rectangle(
                        self.clone, 
                        (scaled_x, scaled_y), 
                        (scaled_x + scaled_size, scaled_y + scaled_size), 
                        (0, 0, 255), 1
                    )
    
    def run_interactive_selection(self):
        """
        Run the interactive polygon selection interface.
        
        Instructions are displayed to guide user interaction.
        """
        if self.thumbnail_image is None:
            print("Error: No thumbnail loaded. Please generate thumbnail first.")
            return False
        
        print("\n" + "="*60)
        print("WSI GRID SELECTOR - INTERACTIVE MODE")
        print("="*60)
        print("Instructions:")
        print("• Left click: Add polygon vertex")
        print("• Right click: Complete polygon and generate grid")
        print("• Press any key: Exit and save results")
        print("• Multiple polygons can be drawn sequentially")
        print("="*60 + "\n")
        
        # Set up OpenCV window and mouse callback
        cv2.imshow("WSI Grid Selector", self.thumbnail_image)
        cv2.setMouseCallback("WSI Grid Selector", self._mouse_callback)
        
        # Wait for user interaction
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
    
    def save_results(self, output_image='output_grid_selection.png', 
                    coordinates_file='grid_coordinates.h5'):
        """
        Save the final annotated image and grid coordinates.
        
        Args:
            output_image (str): Name of output image file
            coordinates_file (str): Name of HDF5 coordinates file
        """
        try:
            # Save annotated image
            output_image_path = os.path.join(self.output_dir, output_image)
            cv2.imwrite(output_image_path, self.clone)
            print(f"Annotated image saved as: {output_image}")
            
            # Save coordinates to HDF5 file
            coordinates_path = os.path.join(self.output_dir, coordinates_file)
            with h5py.File(coordinates_path, "w") as h5file:
                # Convert coordinates to array format [x1, y1, x2, y2]
                coordinates_array = np.array(
                    self.grid_coordinates, dtype=np.int32
                ).reshape(-1, 4)
                
                h5file.create_dataset("coordinates", data=coordinates_array)
                
                # Save metadata
                h5file.attrs['scale_factor'] = self.scale_factor
                h5file.attrs['rect_size'] = self.rect_size
                h5file.attrs['total_grids'] = len(self.grid_coordinates)
                h5file.attrs['source_file'] = os.path.basename(self.svs_file_path)
            
            print(f"Grid coordinates saved to: {coordinates_file}")
            print(f"Total grid squares generated: {len(self.grid_coordinates)}")
            
            return True
            
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def get_statistics(self):
        """
        Print statistics about the generated grid.
        """
        if not self.grid_coordinates:
            print("No grid coordinates generated yet.")
            return
        
        print("\n" + "="*40)
        print("GRID GENERATION STATISTICS")
        print("="*40)
        print(f"Source file: {os.path.basename(self.svs_file_path)}")
        print(f"Scale factor: {self.scale_factor}")
        print(f"Grid size: {self.rect_size} x {self.rect_size} pixels")
        print(f"Total grids: {len(self.grid_coordinates)}")
        print(f"Coverage area: {len(self.grid_coordinates) * (self.rect_size ** 2):,} pixels²")
        print("="*40 + "\n")


def main():
    """
    Main function to run the WSI Grid Selector.
    
    Modify the parameters below according to your needs.
    """
    # Configuration parameters
    SVS_FILE = 'TCGA-BR-6709-01Z-00-DX1.92df4063-8b47-4655-a010-edc385b35840.svs'
    OUTPUT_DIR = '.'
    SCALE_FACTOR = 100
    
    # Verify input file exists
    if not os.path.exists(SVS_FILE):
        print(f"Error: SVS file '{SVS_FILE}' not found!")
        print("Please update the SVS_FILE path in the main() function.")
        return
    
    # Initialize the grid selector
    selector = WSIGridSelector(SVS_FILE, OUTPUT_DIR, SCALE_FACTOR)
    
    # Load slide and generate thumbnail
    if not selector.load_slide():
        return
    
    if not selector.generate_thumbnail('thumbnail.png'):
        return
    
    # Run interactive selection
    if selector.run_interactive_selection():
        # Save results
        selector.save_results()
        selector.get_statistics()
    
    print("WSI Grid Selector completed successfully!")


if __name__ == "__main__":
    main()
