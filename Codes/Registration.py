import SimpleITK as sitk

def est_lin_transf(im_ref, im_mov, mask = None, verbose = False):
    """
    Estimate linear (affine) transform to align `im_mov` to `im_ref`
    and return the transform parameters.
    """
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings
    registration_method.SetMetricAsMeanSquares()  # Can use other metrics like Mattes Mutual Information
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)  # Use a small percentage of pixels for speed
    
    if mask is not None:
        registration_method.SetMetricFixedMask(mask)
        
    # Interpolation
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=500, convergenceMinimumValue=1e-3, convergenceWindowSize=20)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Transformation model
    initial_transform = sitk.CenteredTransformInitializer(im_ref, im_mov, sitk.AffineTransform(im_ref.GetDimension()), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Perform registration
    final_transform = registration_method.Execute(sitk.Cast(im_ref, sitk.sitkFloat32), 
                                                  sitk.Cast(im_mov, sitk.sitkFloat32))

    if verbose:
        print("--------")
        print("Affine registration:")
        print(f'Final metric value: {registration_method.GetMetricValue()}')
        print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        print("--------")
    
    return final_transform

def est_nl_transf(im_ref, im_mov, mask = None, verbose = False):
    """
    Estimate non-linear (BSpline) transform to align `im_mov` to `im_ref`
    and return the transform parameters.
    """
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    if mask is not None:
        registration_method.SetMetricFixedMask(mask)

    # Interpolation
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=500, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=500)
    
    # Transformation model (BSpline)
    grid_physical_spacing = [50.0] * im_ref.GetDimension()  # Grid spacing in physical units
    image_spacing = im_ref.GetSpacing()
    image_size = im_ref.GetSize()
    grid_size = [int(image_size[i] * image_spacing[i] / grid_physical_spacing[i] + 0.5) for i in range(im_ref.GetDimension())]
    grid_size = [max(3, grid_size[i]) for i in range(len(grid_size))]  # Ensure grid size is at least 3x3
    transform_domain_mesh_size = [g - 1 for g in grid_size]
    initial_transform = sitk.BSplineTransformInitializer(im_ref, transform_domain_mesh_size)
    registration_method.SetInitialTransformAsBSpline(initial_transform, inPlace=False)
    
    # Perform registration
    final_transform = registration_method.Execute(im_ref, im_mov)

    if verbose:
        print("--------")
        print("Non-linear registration:")
        print(f'Final metric value: {registration_method.GetMetricValue()}')
        print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        print(f"Final Iteration: {registration_method.GetOptimizerIteration()}")
        print("--------")
    
    return final_transform

def apply_lin_transf(im_mov, lin_xfm):
    """
    Apply given linear transform `lin_xfm` to `im_mov` and return the transformed image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_mov)  # Use the moving image as reference
    resampler.SetTransform(lin_xfm)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    transformed_image = resampler.Execute(im_mov)
    return transformed_image

def apply_nl_transf(im_mov, nl_xfm):
    """
    Apply given non-linear transform `nl_xfm` to `im_mov` and return the transformed image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_mov)
    resampler.SetTransform(nl_xfm)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    transformed_image = resampler.Execute(im_mov)
    return transformed_image
