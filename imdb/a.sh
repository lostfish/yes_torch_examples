echo $1
grep train $1|awk -F: 'BEGIN{a=0}{if($NF > a){a=$NF;print}}'|tail -n1
grep valid $1|awk -F: 'BEGIN{a=0}{if($NF > a){a=$NF;print}}'|tail -n1
sed 's/^.*time://' $1|awk 'BEGIN{a=0}{a+=$1}END{print "time:",a/60,NR}'
echo "------"
