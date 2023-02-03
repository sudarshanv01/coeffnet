# Check if a folder called `test_data`, `train_data` or `validation_data`
# exists in the current directory and then delete it
if [ -d test_data ]; then
    rm -r test_data
fi
if [ -d datasets_checkpoint/train_data ]; then
    rm -r train_data
fi
if [ -d datasets_chkpoint/validation_data ]; then
    rm -r datasets_chkpoint/validation_data
fi