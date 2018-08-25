import os

folders = os.listdir("images/")

for folder in folders:
    files = [int(item.replace('.jpg', '')) for item in os.listdir("images/" + folder + "/")]
    files.sort()
    files = [str(item).zfill(6)+"\n" for item in files]
    fileString = "".join(files)

    set_file = open(folder + "_set_filename.txt", "w")
    set_file.write(fileString)
    set_file.close()