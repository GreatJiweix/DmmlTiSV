#Folder_A="/data1/xjw/vox2_train"
Folder_A="/data1/xjw/vox2"  
for file_a in ${Folder_A}/*; do  
    temp_file=`basename $file_a`  
    echo $file_a
    for file_b in ${file_a}/*; do
        temp_file1=`basename $file_b`
        for file_c in ${file_b}/*; do
        echo $file_c
        #if [ "${file_c##*.}" = "m4a" ]; then
        avconv -i "$file_c" "${file_c/%m4a/wav}"
        rm -rf $file_c
    #for file in ${temp_file}/*; do
        #temp2_file=`basename $file`
        #avconv -i "$temp2_file" "${temp2_file/%m4a/wav}"
        #echo $temp2_file
done  
done
done
fi