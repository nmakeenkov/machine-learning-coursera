for i in "$@"
do
	tr -d '\n' < $i > temp.txt && mv temp.txt $i
done
