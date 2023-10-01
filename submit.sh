git lfs track my_submission/model.pth
git add my_submission/model.ckpt
git add .
git commit -m "convnext large xx 0.62"
version=49

# Create a tag for your submission and push
git tag -am "submission-v0.${version}" submission-v0.${version}
git push aicrowd master
git push aicrowd submission-v0.${version}