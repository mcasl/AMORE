\name{select.activation.function}
\alias{select.activation.function}
\title{Provides R code of the selected activation function.}
\description{Provides random values to the neuron weights and bias so as to start with. It is usually called by the random.init.NeuralNet function during the construction of the neural object by the \emph{newff} function. 
}
\usage{ select.activation.function(activation.function)}
\arguments{
\item{activation.function}{ Activation function name. Currently the user may choose amongst \emph{purelin}, \emph{tansig}, \emph{sigmoid}, \emph{hardlim} and \emph{custom}. If \emph{custom} is chosen the the user must manually assign the neuron  \emph{f0} and \emph{f1} functions.}
 }
\value{\emph{select.activation.function} returns a list with two elements. The first, \emph{f0} is the R code selected to serve as the neuron activation function. The second, \emph{f1} is the R code of the activation function derivative.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré.	            \email{j.ordieres@upm.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unirioja.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{fjmartin@unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}
\seealso{
\code{\link{init.MLPneuron}}, \code{\link{newff}}
}
\keyword{neural}
