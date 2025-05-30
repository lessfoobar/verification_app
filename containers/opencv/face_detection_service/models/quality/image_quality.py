#!/usr/bin/env python3
"""
Image Quality Analysis Implementation
====================================

Extracted from face_detection.py lines 338-385
Comprehensive image/video quality analysis for verification.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

from ..base import QualityModel, ModelInfo, measure_inference_time, ensure_loaded, validate_input_decorator
from ...core.data_structures import QualityMetrics
from ...core.exceptions import ModelLoadError, QualityAnalysisError, PoorQualityError
from ...core.constants import (
    MIN_BRIGHTNESS_SCORE,
    MIN_SHARPNESS_SCORE,
    MIN_OVERALL_QUALITY,
    MIN_FACE_SIZE_RATIO,
    LAPLACIAN_SHARPNESS_THRESHOLD,
    TEXTURE_VARIANCE_THRESHOLD,
    COLOR_VARIANCE_THRESHOLD,
    OPTIMAL_BRIGHTNESS_VALUE,
    OPTIMAL_BRIGHTNESS_RANGE,
    MODEL_VERSIONS
)


class ImageQualityAnalyzer(QualityModel):
    """Comprehensive image quality analysis implementation"""
    
    def __init__(self, name: str = "image_quality", config: Optional[Dict[str, Any]] = None):
        """Initialize image quality analyzer"""
        default_config = {
            'min_brightness_score': MIN_BRIGHTNESS_SCORE,
            'min_sharpness_score': MIN_SHARPNESS_SCORE,
            'min_overall_quality': MIN_OVERALL_QUALITY,
            'min_face_size_ratio': MIN_FACE_SIZE_RATIO,
            'laplacian_threshold': LAPLACIAN_SHARPNESS_THRESHOLD,
            'texture_threshold': TEXTURE_VARIANCE_THRESHOLD,
            'color_threshold': COLOR_VARIANCE_THRESHOLD,
            'optimal_brightness': OPTIMAL_BRIGHTNESS_VALUE,
            'brightness_range': OPTIMAL_BRIGHTNESS_RANGE,
            'enable_advanced_metrics': True,
            'enable_face_specific_analysis': True,
            'enabled': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        # Configuration
        self.min_brightness_score = self.config['min_brightness_score']
        self.min_sharpness_score = self.config['min_sharpness_score']
        self.min_overall_quality = self.config['min_overall_quality']
        self.min_face_size_ratio = self.config['min_face_size_ratio']
        self.laplacian_threshold = self.config['laplacian_threshold']
        self.texture_threshold = self.config['texture_threshold']
        self.color_threshold = self.config['color_threshold']
        self.optimal_brightness = self.config['optimal_brightness']
        self.brightness_range = self.config['brightness_range']
        self.enable_advanced_metrics = self.config['enable_advanced_metrics']
        self.enable_face_specific_analysis = self.config['enable_face_specific_analysis']
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def load_model(self) -> None:
        """Load quality analysis model (mainly configuration validation)"""
        start_time = time.time()
        
        try:
            self.logger.info("Initializing Image Quality Analyzer...")
            
            # Validate configuration
            self._validate_config()
            
            # Test OpenCV functionality
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            _ = cv2.Laplacian(test_image, cv2.CV_64F)
            
            self._load_time = (time.time() - start_time) * 1000
            self._is_loaded = True
            
            self.logger.info(f"✅ Image Quality Analyzer loaded successfully ({self._load_time:.1f}ms)")
            
        except Exception as e:
            self._is_loaded = False
            error_msg = f"Failed to load image quality analyzer: {e}"
            self.logger.error(error_msg)
            raise ModelLoadError(self.name, error_msg)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not 0.0 <= self.min_brightness_score <= 1.0:
            raise ValueError("min_brightness_score must be between 0.0 and 1.0")
        
        if not 0.0 <= self.min_sharpness_score <= 1.0:
            raise ValueError("min_sharpness_score must be between 0.0 and 1.0")
        
        if not 0.0 <= self.min_overall_quality <= 1.0:
            raise ValueError("min_overall_quality must be between 0.0 and 1.0")
        
        if self.laplacian_threshold < 0:
            raise ValueError("laplacian_threshold must be non-negative")
    
    def unload_model(self) -> None:
        """Unload quality analyzer (cleanup)"""
        self._is_loaded = False
        self.logger.info("Image Quality Analyzer unloaded")
    
    def is_loaded(self) -> bool:
        """Check if analyzer is ready"""
        return self._is_loaded
    
    def get_info(self) -> ModelInfo:
        """Get analyzer information"""
        if self._model_info is None:
            self._model_info = ModelInfo(
                name=self.name,
                version=MODEL_VERSIONS.get('opencv', 'unknown'),
                provider="OpenCV + Custom Analysis",
                capabilities=[
                    'brightness_analysis',
                    'sharpness_analysis',
                    'contrast_analysis',
                    'noise_analysis',
                    'face_quality_analysis',
                    'composition_analysis',
                    'color_analysis',
                    'texture_analysis'
                ],
                input_formats=['BGR', 'RGB', 'grayscale', 'numpy_array'],
                output_format='QualityMetrics',
                metadata={
                    'advanced_metrics': self.enable_advanced_metrics,
                    'face_specific_analysis': self.enable_face_specific_analysis,
                    'thresholds': {
                        'brightness': self.min_brightness_score,
                        'sharpness': self.min_sharpness_score,
                        'overall': self.min_overall_quality,
                        'face_size': self.min_face_size_ratio
                    }
                }
            )
        
        return self._model_info
    
    def get_quality_requirements(self) -> Dict[str, float]:
        """Get minimum quality requirements"""
        return {
            'brightness_score': self.min_brightness_score,
            'sharpness_score': self.min_sharpness_score,
            'overall_quality': self.min_overall_quality,
            'face_size_ratio': self.min_face_size_ratio
        }
    
    @ensure_loaded
    @validate_input_decorator
    @measure_inference_time
    def analyze_quality(self, image: np.ndarray, face_regions: Optional[List[List[int]]] = None) -> QualityMetrics:
        """
        Analyze image quality with comprehensive metrics
        
        Args:
            image: Input image (BGR or grayscale)
            face_regions: Optional face bounding boxes for focused analysis
            
        Returns:
            QualityMetrics with quality analysis
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # For face analysis
            
            # Basic quality metrics
            brightness_score = self._analyze_brightness(gray)
            sharpness_score = self._analyze_sharpness(gray)
            
            # Face-specific analysis
            face_size_ratio = 0.0
            stability_score = 0.8  # Default for single frame
            
            if face_regions:
                face_size_ratio = self._analyze_face_size(image, face_regions)
                
                if self.enable_face_specific_analysis:
                    # Enhanced analysis with face regions
                    face_quality_scores = []
                    for bbox in face_regions:
                        face_score = self._analyze_face_region_quality(image, bbox)
                        face_quality_scores.append(face_score)
                    
                    if face_quality_scores:
                        # Use average face quality to adjust overall scores
                        avg_face_quality = np.mean(face_quality_scores)
                        sharpness_score = (sharpness_score + avg_face_quality) / 2
            
            # Advanced metrics if enabled
            additional_metrics = {}
            if self.enable_advanced_metrics:
                additional_metrics = self._compute_advanced_metrics(image, gray, face_regions)
            
            # Calculate overall quality score
            overall_quality = self._calculate_overall_quality(
                brightness_score, sharpness_score, face_size_ratio, stability_score, additional_metrics
            )
            
            result = QualityMetrics(
                brightness_score=brightness_score,
                sharpness_score=sharpness_score,
                face_size_ratio=face_size_ratio,
                stability_score=stability_score,
                overall_quality=overall_quality,
                additional_metrics=additional_metrics
            )
            
            self.logger.debug(
                f"Quality analysis: overall={overall_quality:.3f}, "
                f"brightness={brightness_score:.3f}, sharpness={sharpness_score:.3f}, "
                f"face_size={face_size_ratio:.3f}"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Quality analysis failed: {e}"
            self.logger.error(error_msg)
            raise QualityAnalysisError(error_msg, "overall")
    
    def _analyze_brightness(self, gray_image: np.ndarray) -> float:
        """Analyze image brightness"""
        try:
            # Calculate mean brightness
            brightness = np.mean(gray_image)
            
            # Score based on distance from optimal brightness
            brightness_score = 1.0 - abs(brightness - self.optimal_brightness) / 128
            
            # Ensure score is within valid range
            brightness_score = max(0.0, min(brightness_score, 1.0))
            
            return brightness_score
            
        except Exception as e:
            self.logger.warning(f"Brightness analysis failed: {e}")
            return 0.0
    
    def _analyze_sharpness(self, gray_image: np.ndarray) -> float:
        """Analyze image sharpness using Laplacian variance"""
        try:
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            laplacian_var = laplacian.var()
            
            # Normalize to 0-1 scale
            sharpness_score = min(laplacian_var / self.laplacian_threshold, 1.0)
            
            return max(0.0, sharpness_score)
            
        except Exception as e:
            self.logger.warning(f"Sharpness analysis failed: {e}")
            return 0.0
    
    def _analyze_face_size(self, image: np.ndarray, face_regions: List[List[int]]) -> float:
        """Analyze face size relative to image"""
        try:
            if not face_regions:
                return 0.0
            
            image_area = image.shape[0] * image.shape[1]
            
            # Calculate total face area
            total_face_area = 0
            for bbox in face_regions:
                x, y, w, h = bbox
                face_area = w * h
                total_face_area += face_area
            
            # Calculate ratio
            face_size_ratio = total_face_area / image_area
            
            return min(face_size_ratio, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Face size analysis failed: {e}")
            return 0.0
    
    def _analyze_face_region_quality(self, image: np.ndarray, bbox: List[int]) -> float:
        """Analyze quality of specific face region"""
        try:
            x, y, w, h = bbox
            
            # Extract face region
            face_region = image[max(0, y):min(image.shape[0], y + h),
                              max(0, x):min(image.shape[1], x + w)]
            
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Analyze face-specific quality metrics
            face_brightness = self._analyze_brightness(face_gray)
            face_sharpness = self._analyze_sharpness(face_gray)
            
            # Face-specific metrics
            contrast_score = self._analyze_contrast(face_gray)
            texture_score = self._analyze_texture(face_gray)
            
            # Combine scores
            face_quality = (
                face_brightness * 0.3 +
                face_sharpness * 0.4 +
                contrast_score * 0.2 +
                texture_score * 0.1
            )
            
            return max(0.0, min(face_quality, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Face region quality analysis failed: {e}")
            return 0.0
    
    def _analyze_contrast(self, gray_image: np.ndarray) -> float:
        """Analyze image contrast"""
        try:
            # Calculate standard deviation as contrast measure
            contrast = np.std(gray_image)
            
            # Normalize to 0-1 scale (assuming max std ~50 for good contrast)
            contrast_score = min(contrast / 50.0, 1.0)
            
            return max(0.0, contrast_score)
            
        except Exception as e:
            self.logger.warning(f"Contrast analysis failed: {e}")
            return 0.0
    
    def _analyze_texture(self, gray_image: np.ndarray) -> float:
        """Analyze image texture complexity"""
        try:
            # Use Sobel operators to detect edges/texture
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            texture_score = np.mean(gradient_magnitude)
            
            # Normalize to 0-1 scale
            texture_score = min(texture_score / self.texture_threshold, 1.0)
            
            return max(0.0, texture_score)
            
        except Exception as e:
            self.logger.warning(f"Texture analysis failed: {e}")
            return 0.0
    
    def _compute_advanced_metrics(self, image: np.ndarray, gray_image: np.ndarray, face_regions: Optional[List[List[int]]]) -> Dict[str, float]:
        """Compute advanced quality metrics"""
        advanced_metrics = {}
        
        try:
            # Noise analysis
            advanced_metrics['noise_level'] = self._analyze_noise(gray_image)
            
            # Color analysis (if color image)
            if len(image.shape) == 3:
                advanced_metrics['color_variance'] = self._analyze_color_variance(image)
                advanced_metrics['color_saturation'] = self._analyze_color_saturation(image)
            
            # Composition analysis
            advanced_metrics['composition_score'] = self._analyze_composition(gray_image, face_regions)
            
            # Exposure analysis
            advanced_metrics['exposure_score'] = self._analyze_exposure(gray_image)
            
            # Focus analysis (different from sharpness)
            advanced_metrics['focus_score'] = self._analyze_focus(gray_image)
            
        except Exception as e:
            self.logger.warning(f"Advanced metrics computation failed: {e}")
        
        return advanced_metrics
    
    def _analyze_noise(self, gray_image: np.ndarray) -> float:
        """Analyze image noise level"""
        try:
            # Use bilateral filter to separate noise from edges
            filtered = cv2.bilateralFilter(gray_image, 9, 75, 75)
            noise = cv2.absdiff(gray_image, filtered)
            noise_level = np.mean(noise)
            
            # Convert to quality score (lower noise = higher score)
            noise_score = max(0.0, 1.0 - (noise_level / 50.0))
            
            return noise_score
            
        except Exception as e:
            self.logger.warning(f"Noise analysis failed: {e}")
            return 0.5
    
    def _analyze_color_variance(self, image: np.ndarray) -> float:
        """Analyze color variance"""
        try:
            # Calculate variance for each color channel
            variances = [np.var(image[:, :, i]) for i in range(3)]
            avg_variance = np.mean(variances)
            
            # Normalize to 0-1 scale
            color_variance_score = min(avg_variance / self.color_threshold, 1.0)
            
            return max(0.0, color_variance_score)
            
        except Exception as e:
            self.logger.warning(f"Color variance analysis failed: {e}")
            return 0.5
    
    def _analyze_color_saturation(self, image: np.ndarray) -> float:
        """Analyze color saturation"""
        try:
            # Convert to HSV and analyze saturation channel
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            
            # Calculate mean saturation
            avg_saturation = np.mean(saturation)
            saturation_score = avg_saturation / 255.0
            
            return max(0.0, min(saturation_score, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Color saturation analysis failed: {e}")
            return 0.5
    
    def _analyze_composition(self, gray_image: np.ndarray, face_regions: Optional[List[List[int]]]) -> float:
        """Analyze image composition"""
        try:
            h, w = gray_image.shape
            
            # If faces are provided, check if they follow rule of thirds
            if face_regions:
                composition_score = 0.0
                
                for bbox in face_regions:
                    x, y, face_w, face_h = bbox
                    
                    # Calculate face center
                    face_center_x = x + face_w // 2
                    face_center_y = y + face_h // 2
                    
                    # Check proximity to rule of thirds lines
                    third_x = w // 3
                    third_y = h // 3
                    
                    # Distance to nearest third line
                    dist_x = min(abs(face_center_x - third_x), abs(face_center_x - 2 * third_x))
                    dist_y = min(abs(face_center_y - third_y), abs(face_center_y - 2 * third_y))
                    
                    # Normalize distances
                    norm_dist_x = dist_x / w
                    norm_dist_y = dist_y / h
                    
                    # Closer to third lines = better composition
                    face_composition = 1.0 - (norm_dist_x + norm_dist_y) / 2
                    composition_score = max(composition_score, face_composition)
                
                return max(0.0, min(composition_score, 1.0))
            
            # If no faces, analyze general composition
            # Use variance in spatial distribution
            mean_intensity = np.mean(gray_image)
            spatial_variance = np.var(gray_image)
            
            # Higher spatial variance often indicates better composition
            composition_score = min(spatial_variance / 1000.0, 1.0)
            
            return max(0.0, composition_score)
            
        except Exception as e:
            self.logger.warning(f"Composition analysis failed: {e}")
            return 0.5
    
    def _analyze_exposure(self, gray_image: np.ndarray) -> float:
        """Analyze image exposure"""
        try:
            # Calculate histogram
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            
            # Check for clipping (overexposure/underexposure)
            total_pixels = gray_image.shape[0] * gray_image.shape[1]
            
            # Pixels in shadows (0-16) and highlights (240-255)
            shadows = np.sum(hist[0:17]) / total_pixels
            highlights = np.sum(hist[240:256]) / total_pixels
            
            # Good exposure has minimal clipping
            clipping_penalty = (shadows + highlights) * 2
            exposure_score = max(0.0, 1.0 - clipping_penalty)
            
            return exposure_score
            
        except Exception as e:
            self.logger.warning(f"Exposure analysis failed: {e}")
            return 0.5
    
    def _analyze_focus(self, gray_image: np.ndarray) -> float:
        """Analyze image focus using frequency domain"""
        try:
            # Use FFT to analyze frequency content
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Calculate high frequency content
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            
            # Create mask for high frequencies (outer region)
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x)**2 + (y - center_y)**2) > (min(h, w) // 4)**2
            
            high_freq_energy = np.sum(magnitude_spectrum[mask])
            total_energy = np.sum(magnitude_spectrum)
            
            # Focus score based on high frequency content
            focus_score = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # Normalize to reasonable range
            focus_score = min(focus_score * 10, 1.0)
            
            return max(0.0, focus_score)
            
        except Exception as e:
            self.logger.warning(f"Focus analysis failed: {e}")
            return 0.5
    
    def _calculate_overall_quality(self, brightness_score: float, sharpness_score: float,
                                  face_size_ratio: float, stability_score: float,
                                  additional_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        try:
            # Base quality from main metrics
            base_quality = (
                brightness_score * 0.25 +
                sharpness_score * 0.35 +
                min(face_size_ratio * 5, 1.0) * 0.25 +  # Scale face size importance
                stability_score * 0.15
            )
            
            # Adjust with advanced metrics if available
            if additional_metrics:
                adjustments = 0.0
                adjustment_count = 0
                
                # Positive adjustments
                for metric in ['focus_score', 'composition_score', 'exposure_score']:
                    if metric in additional_metrics:
                        adjustments += additional_metrics[metric] * 0.1
                        adjustment_count += 1
                
                # Negative adjustments
                if 'noise_level' in additional_metrics:
                    adjustments += (1.0 - additional_metrics['noise_level']) * 0.05
                    adjustment_count += 1
                
                if adjustment_count > 0:
                    avg_adjustment = adjustments / adjustment_count
                    base_quality = (base_quality * 0.8) + (avg_adjustment * 0.2)
            
            return max(0.0, min(base_quality, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Overall quality calculation failed: {e}")
            return 0.0
    
    def validate_quality_threshold(self, quality_metrics: QualityMetrics) -> bool:
        """Validate that quality meets minimum thresholds"""
        return quality_metrics.is_acceptable
    
    def get_quality_feedback(self, quality_metrics: QualityMetrics) -> List[str]:
        """Get user-friendly quality feedback"""
        issues = quality_metrics.get_quality_issues()
        
        feedback_map = {
            'poor_lighting': "Improve lighting conditions",
            'blurry_image': "Ensure camera is in focus",
            'face_too_small': "Move closer to camera",
            'unstable_video': "Keep device steady"
        }
        
        feedback = []
        for issue in issues:
            if issue in feedback_map:
                feedback.append(feedback_map[issue])
        
        if not feedback:
            feedback.append("Quality is acceptable")
        
        return feedback
    
    def batch_analyze_quality(self, images: List[np.ndarray], face_regions_list: Optional[List[List[List[int]]]] = None) -> List[QualityMetrics]:
        """Analyze quality for multiple images"""
        results = []
        face_regions_list = face_regions_list or [None] * len(images)
        
        for i, (image, face_regions) in enumerate(zip(images, face_regions_list)):
            try:
                result = self.analyze_quality(image, face_regions)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch quality analysis failed for image {i}: {e}")
                # Add default low-quality result
                results.append(QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0))
        
        return results


# Factory registration
from ..base import ModelFactory
ModelFactory.register('image_quality', ImageQualityAnalyzer)


# Utility functions
def create_image_quality_analyzer(config: Optional[Dict[str, Any]] = None) -> ImageQualityAnalyzer:
    """Create image quality analyzer with default configuration"""
    return ImageQualityAnalyzer("image_quality", config)


# Example usage and testing
if __name__ == '__main__':
    print("📊 Image Quality Analysis Model")
    print("=" * 40)
    
    # Create analyzer
    analyzer = create_image_quality_analyzer({
        'enable_advanced_metrics': True,
        'enable_face_specific_analysis': True
    })
    
    print(f"🔧 Analyzer configuration: {analyzer.config}")
    
    # Load model
    try:
        analyzer.load_model()
        print("✅ Analyzer loaded successfully")
        
        # Get model info
        info = analyzer.get_info()
        print(f"📋 Model info:")
        print(f"   Name: {info.name}")
        print(f"   Provider: {info.provider}")
        print(f"   Capabilities: {info.capabilities}")
        
        # Test with synthetic image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add some features to make it more realistic
        cv2.ellipse(test_image, (320, 240), (60, 80), 0, 0, 360, (200, 180, 160), -1)
        cv2.rectangle(test_image, (100, 100), (200, 200), (150, 150, 150), -1)
        
        # Add noise
        noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
        test_image = cv2.addWeighted(test_image, 0.9, noise, 0.1, 0)
        
        # Test quality analysis without face regions
        result = analyzer.analyze_quality(test_image)
        print(f"🔍 Test quality analysis (no faces):")
        print(f"   Overall quality: {result.overall_quality:.3f}")
        print(f"   Brightness score: {result.brightness_score:.3f}")
        print(f"   Sharpness score: {result.sharpness_score:.3f}")
        print(f"   Face size ratio: {result.face_size_ratio:.3f}")
        print(f"   Is acceptable: {result.is_acceptable}")
        
        # Test with face regions
        face_regions = [[200, 160, 120, 160]]  # Simulated face bbox
        result_with_faces = analyzer.analyze_quality(test_image, face_regions)
        print(f"🔍 Test quality analysis (with faces):")
        print(f"   Overall quality: {result_with_faces.overall_quality:.3f}")
        print(f"   Brightness score: {result_with_faces.brightness_score:.3f}")
        print(f"   Sharpness score: {result_with_faces.sharpness_score:.3f}")
        print(f"   Face size ratio: {result_with_faces.face_size_ratio:.3f}")
        print(f"   Is acceptable: {result_with_faces.is_acceptable}")
        
        # Test advanced metrics
        if result_with_faces.additional_metrics:
            print(f"📊 Advanced metrics:")
            for metric, value in result_with_faces.additional_metrics.items():
                print(f"   {metric}: {value:.3f}")
        
        # Test quality feedback
        feedback = analyzer.get_quality_feedback(result_with_faces)
        print(f"💡 Quality feedback: {feedback}")
        
        # Test quality validation
        meets_threshold = analyzer.validate_quality_threshold(result_with_faces)
        print(f"✅ Meets quality threshold: {meets_threshold}")
        
        # Test performance
        performance = analyzer.get_performance()
        if performance:
            print(f"⚡ Performance:")
            print(f"   Load time: {performance.load_time_ms:.1f}ms")
            print(f"   Inference time: {performance.inference_time_ms:.1f}ms")
        
        # Test batch processing
        test_images = [test_image, test_image.copy(), test_image.copy()]
        batch_results = analyzer.batch_analyze_quality(test_images)
        print(f"📦 Batch processing:")
        print(f"   Processed {len(batch_results)} images")
        print(f"   Average quality: {np.mean([r.overall_quality for r in batch_results]):.3f}")
        
        # Test requirements
        requirements = analyzer.get_quality_requirements()
        print(f"📋 Quality requirements: {requirements}")
        
        # Cleanup
        analyzer.unload_model()
        print("✅ Analyzer unloaded")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Image quality analysis test completed")