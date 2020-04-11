from collections import deque

d = deque()
d.appendleft(1)
d.appendleft(2)

l = []
l.extend(d)

print l
print d[0]

empty_d = deque()

if not empty_d:
    print "here"


import numpy as np
matrix = np.array( [[0, 0, 1], 
                    [0, 0, 0]])
print np.nonzero(matrix)

print int(False)

arr1, arr2 = matrix 
print arr1

print np.array([1, 2])**5

print 3**0.5


print np.random.choice(xrange(4), size=(3,), replace=False)[[1,2]]

print np.stack([np.array([1,2]), np.array([4,5])])

print ""
mul = np.matmul(np.array([1,2,3]).reshape(-1,1),np.array([[1,2,3]]).reshape(1,-1))
print mul
print np.sum(mul, axis=0)

print "\n\n"
print np.unique([[1, 1, 5, 5],
                 [2, 2, 1, 5]], axis=1)

print "\n\n"
print np.unique([1, 1, 5, 5])

for x in {1: "a", 2: "b"}.iteritems():
    print x


from collections import Counter
print Counter(["a", "a", "a", "b"]).most_common(1)[0][0]

print [1,2,3][:100]

# from sys import stdout
# from time import sleep

# for i in xrange(100):
#     stdout.write("\rlol"+str(i))
#     stdout.flush()
#     sleep(0.01)

# print ""
# print "here"



# from PIL import Image
# img = Image.open("./testimg.jpg")
# numyimgflippped = np.fliplr(np.array(img))
# new_img = Image.fromarray(numyimgflippped, 'RGB')
# new_img.show()
print ""
arr1 = np.array([[[1,2],[1,2],[1,2]]])
print arr1.shape
arr2 = np.array([[[1,2],[1,2],[1,2]]]).T
print arr2.shape
print np.squeeze(np.tensordot(arr1,arr2,axes=[[2],[0]]))

print np.reshape(np.array([1,2]), (-1,1))

print ""
x = [[1, 2, 3],
     [4, 5, 6]]
x = np.array(x)

y = np.reshape(x, (2,1,3))
z = np.reshape(x, (1,2,3))
print "-----"
print np.squeeze(np.tensordot(y,z,axes=[[2],[2]]))

print np.concatenate([np.array([[1,2,3],[3,4,5]]), np.array([[6,7,8],[9,10,11]])])

print int(-1>=0)

print np.dot(np.array([1.0,2]), np.array([1,2]))

print np.exp(4)

print "----"
def test(i,j):
    print("i:", i)
    print("j:", j)
    return i+j
print np.fromfunction(lambda i, j: test(i,j), shape=(2,2))

A = np.reshape(np.array([1, 2, 3]), (3))
B = np.reshape(np.array([1, 1, 1]), (3,1))

print np.squeeze(np.matmul(A, B))



X = np.array([[0, 1], 
              [1, 2],
              [1, 2]])
Y = np.array([2, 4, 2])
print X[np.where(Y==2)[0]]

print float("-inf")

# import tensorflow as tf
# a_tensor = tf.constant([4, 1, 4])
# b_tensor = tf.constant([0, 1, 0])
# sess = tf.Session()
# print sess.run(tf.equal(a_tensor, b_tensor))
# print sess.run(tf.reduce_mean(tf.cast(tf.equal(a_tensor, b_tensor), tf.float32)))


