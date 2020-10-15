f1 = open('epoch_loss_img_gen_delta.txt', 'r')
s = f1.read().split()
f2 = open('epoch_loss_img_gen_delta1.txt', 'w')
for i in range(len(s)):
    f2.write(f"{s[i]} ")
    if i%3 == 2:
        f2.write("\n")
f2.close()
f1.close()