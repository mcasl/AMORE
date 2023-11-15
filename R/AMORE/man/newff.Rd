\name{newff}
\alias{newff}
\title{Create a Multilayer Feedforward Neural Network}
\description{Creates a feedforward artificial neural network according to the structure established by the AMORE package standard.}
\usage{
newff(n.neurons, learning.rate.global, momentum.global, error.criterium, Stao, 
	hidden.layer, output.layer, method) 
}
\arguments{
\item{n.neurons}{Numeric vector containing the number of neurons of each layer. The first element of the vector is the number of input neurons, the last is the number of output neurons and the rest are the number of neuron of the different hidden layers.}
\item{learning.rate.global}{Learning rate at which every neuron is trained.}
\item{momentum.global}{Momentum for every neuron. Needed by several training methods.}
\item{error.criterium}{Criterium used to measure to proximity of the neural network prediction to its target. Currently we can choose amongst: 
\itemize{
\item "LMS": Least Mean Squares.
\item "LMLS": Least Mean Logarithm Squared (Liano 1996).
\item "TAO": TAO Error (Pernia, 2004).
}}
\item{Stao}{Stao parameter for the TAO error criterium. Unused by the rest of criteria.}
\item{hidden.layer}{Activation function of the hidden layer neurons. Available functions are:
\itemize{
\item "purelin".
\item "tansig". 
\item "sigmoid".
\item "hardlim".
\item "custom": The user must manually define the f0 and f1 elements of the neurons.
}}
\item{output.layer}{Activation function of the hidden layer neurons according to the former list shown above.}
\item{method}{Prefered training method. Currently it can be:
\itemize{
\item "ADAPTgd": Adaptative gradient descend.
\item "ADAPTgdwm": Adaptative gradient descend with momentum. 
\item "BATCHgd": BATCH gradient descend.
\item "BATCHgdwm": BATCH gradient descend with momentum.
}}
}
\value{\emph{newff} returns a multilayer feedforward neural network object.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré.	            
Ana González Marcos.                
Alpha V. Pernía Espinoza.           
Eliseo P. Vergara Gonzalez.         
Francisco Javier Martinez de Pisón. 
Fernando Alba Elías.                
}

\references{
Pernía Espinoza, A.V., Ordieres Meré, J.B., Martínez de Pisón, F.J., González Marcos, A. TAO-robust backpropagation learning algorithm. Neural Networks. Vol. 18, Issue 2, pp. 191--204, 2005.\cr \cr
Simon Haykin. Neural Networks -- a Comprehensive Foundation. Prentice Hall, New Jersey, 2nd edition, 1999. ISBN 0-13-273350-1. \cr \cr
}

\seealso{
\code{\link{init.MLPneuron}}, \code{\link{random.init.MLPnet}}, \code{\link{random.init.MLPneuron}}, \code{\link{select.activation.function}} 
}

\examples{
#Example 1

library(AMORE)
# P is the input vector
P <- matrix(sample(seq(-1,1,length=1000), 1000, replace=FALSE), ncol=1) 
# The network will try to approximate the target P^2
target <- P^2                                   
# We create a feedforward network, with two hidden layers.
# The first hidden layer has three neurons and the second has two neurons.
# The hidden layers have got Tansig activation functions and the output layer is Purelin.
net <- newff(n.neurons=c(1,3,2,1), learning.rate.global=1e-2, momentum.global=0.5,
        error.criterium="LMS", Stao=NA, hidden.layer="tansig", 
        output.layer="purelin", method="ADAPTgdwm")
result <- train(net, P, target, error.criterium="LMS", report=TRUE, show.step=100, n.shows=5 )
y <- sim(result$net, P)
plot(P,y, col="blue", pch="+")
points(P,target, col="red", pch="x")
}
\keyword{neural}
