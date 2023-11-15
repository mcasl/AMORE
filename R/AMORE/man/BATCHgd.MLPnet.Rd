\name{BATCHgd.MLPnet}
\alias{BATCHgd.MLPnet}
\title{Batch gradient descent training}
\description{Modifies the neural network weights and biases according to the training set.}
\usage{
BATCHgd.MLPnet(net,P,T,n.epochs, n.threads=0L)
}

\arguments{
\item{net}{Neural Network to train.}
\item{P}{Input data set.}
\item{T}{Target output data set.}
\item{n.epochs}{Number of epochs to train}
\item{n.threads}{Number of threads to spawn. If <1, spawns NumberProcessors-1 threads. If no OpenMP is found, this argument will be ignored.}
}

\value{This function returns a neural network object modified according to the chosen data.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré.	            \email{j.ordieres@upm.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unirioja.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{fjmartin@unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}

\references{
Simon Haykin. Neural Networks -- a Comprehensive Foundation. Prentice Hall, New Jersey, 2nd edition, 1999. ISBN 0-13-273350-1. \cr \cr
}

\seealso{
\code{\link{newff},\link{train},\link{BATCHgdwm.MLPnet}} 
}


\keyword{neural}

