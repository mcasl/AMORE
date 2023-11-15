\name{ADAPTgd.MLPnet}
\alias{ADAPTgd.MLPnet}
\title{Adaptative gradient descent training}
\description{Adaptative gradient descent training method.}
\usage{
ADAPTgd.MLPnet(net,P, T,n.epochs, n.threads=0L)
}

\arguments{
\item{net}{Neural Network to train.}
\item{P}{Input data set.}
\item{T}{Target output data set.}
\item{n.epochs}{Number of epochs to train}
\item{n.threads}{Unused, but required to match the BATCH* function template.}
}

\value{This function returns a neural network object modified according to the input and target data set.}
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
\code{\link{newff},\link{train},\link{ADAPTgdwm.MLPnet}} 
}



\keyword{neural}
