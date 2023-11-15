\name{error.TAO}
\alias{hfun}
\alias{phifun}
\alias{dphifun}

\title{TAO robust error criterium auxiliar functions.}

\description{Auxiliar functions. Not meant to be called from the user but from the \code{\link{error.TAO}} and the \code{\link{deltaE.TAO}} functions.
}

\usage{
hfun(v,k)
phifun(v,k)
dphifun(v,k)
}
\arguments{
\item{v}{Input value.}
\item{k}{Threshold limit.}
}

\value{These functions return a numeric array with dimension equal to the dimension of v.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré.	            \email{j.ordieres@upm.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unirioja.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{fjmartin@unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}

\references{
Pernía Espinoza, A.V., Ordieres Meré, J.B., Martínez de Pisón, F.J., González Marcos, A. TAO-robust backpropagation learning algorithm. Neural Networks. Vol. 18, Issue 2, pp. 191--204, 2005.\cr \cr
Simon Haykin. Neural Networks -- a Comprehensive Foundation. Prentice Hall, New Jersey, 2nd edition, 1999. ISBN 0-13-273350-1. \cr \cr
}

\seealso{
\code{\link{train}}
}

\keyword{neural}

