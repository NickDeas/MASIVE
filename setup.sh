# Create Directories
mkdir models
mkdir models/external
mkdir results
mkdir figures
mkdir external_data

# Make Scripts Executable
chmod +x code/scripts/*

# Download MASIVE
gdown 1MxvcL7M3iGa4-j_hznMTBLXe9ULEAHDf 
echo "Downloaded MASIVE. Untarring and deleting archive..."
tar -xzf masive.tar.gz
rm -f masive.tar.gz

