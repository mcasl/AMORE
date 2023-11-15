library(AMORE)
newff(c(1,50,1), learning.rate.global=0.01, momentum.global=0.01, error.criterium='LMS', hidden.layer='tansig', output.layer='purelin', method='ADAPTgd') -> net
P  <- runif(1000)
T <-  P ^2
system.time( train(net, P, T,n.shows=10,show.step=100) )

# index.show: 10 LMS 5.41332627268925e-06
#    user  system elapsed
#   2.852   0.000   2.853
