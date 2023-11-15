\name{sim.MLPnet}
\alias{sim}
\alias{sim.MLPnet}

\title{Performs the simulation of a neural network from an input data set.}
\description{This function calculates the output values of the neural network for a given data set. Various versions are provided according to different degrees of C code conversion. The \emph{sim.MLPnet} function is the latest and quickest.}

\usage{
sim(net,P,...)
#sim.MLPnet(net,P,...)
}

\arguments{
\item{...}{Currently, the parameters below are accepted.}
\item{net}{Neural Network to simulate.}
\item{P}{Data Set input values.}
}

\value{This function returns a matrix containing the output values of the neural network for the given data set.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré               \email{j.ordieres@upm.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unirioja.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@unirioja.es}\cr
Francisco Javier Martinez de Pisón  \email{fjmartin@unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}

\references{
Simon Haykin. Neural Networks -- a Comprehensive Foundation. Prentice Hall, New Jersey, 2nd edition, 1999. ISBN 0-13-273350-1. \cr \cr
}

\seealso{
\code{\link{newff},\link{train}}
}

\keyword{neural}

