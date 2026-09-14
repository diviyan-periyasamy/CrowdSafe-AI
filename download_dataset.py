from roboflow import Roboflow
rf = Roboflow(api_key="kGme0UqnjFJ3Dol1WmJT")
project = rf.workspace("arbeitsplatz").project("crowd-counting-tvpqo")
version = project.version(1)
dataset = version.download("yolov8")
print("Downloaded to:", dataset.location)