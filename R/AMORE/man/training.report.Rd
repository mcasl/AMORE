\name{training.report}
\alias{training.report}
\title{Neural network training report generator function.}

\description{Function in charge of reporting the behavior of the network training. The users should modify this function according to their needs.}

\usage{
training.report(net,P,T, idx.show, error.criterium)
}
\arguments{
\item{net}{Neural Network to train.}
\item{P}{Training set input values.}
\item{T}{Training set output values}
\item{idx.show}{Current show index.}
\item{error.criterium}{Criterium used to measure the goodness of fit.}
}

\value{This function does not return any value. Just useful for printing and plotting.}
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
\code{\link{train}}
}

\keyword{neural}

