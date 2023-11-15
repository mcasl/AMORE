\name{random.init.MLPneuron}
\alias{random.init.MLPneuron}
\title{Initialize the neuron with random weigths and bias.}
\description{Provides random values to the neuron weights and bias so as to start with. It is usually called by the random.init.NeuralNet function during the construction of the neural object by the \emph{newff} function. 
}
\details{The values are assigned according to the suggestions of \cite{Haykin}.}
\usage{random.init.MLPneuron(net.number.weights, neuron) }
\arguments{
\item{net.number.weights}{Number of bias and weight parameters of the neural network the neuron belongs to.}
\item{neuron}{The neuron object.}
 }
\value{\emph{random.init.MLPneuron} returns the input neuron with bias and weights changed randomly.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré.	            \email{j.ordieres@upm.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unirioja.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{fjmartin@unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}
\seealso{
\code{\link{random.init.MLPnet}}, \code{\link{init.MLPneuron}}, \code{\link{newff}}
}
\references{
Simon Haykin. Neural Networks -- a Comprehensive Foundation. Prentice Hall, New Jersey, 2nd edition, 1999. ISBN 0-13-273350-1. \cr \cr
}
\keyword{neural}
