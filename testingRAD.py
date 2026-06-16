import radiomics
from radiomics import featureextractor

# Initialize the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Print active image types and features to confirm it is fully loaded
print("Active Image Types:", extractor.enabledImagetypes)
print("Active Features:", extractor.enabledFeatures)
