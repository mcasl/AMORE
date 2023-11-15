\name{error.LMS}
\alias{error.LMS}
\alias{error.LMLS}
\alias{error.TAO}
\alias{deltaE.LMS}
\alias{deltaE.LMLS}
\alias{deltaE.TAO}

\title{Neural network training error criteria.}

\description{The error functions calculate the goodness of fit of a neural network according to certain criterium:
\itemize{
\item LMS:  Least Mean Squares Error.
\item LMLS: Least Mean Log Squares minimization. 
\item TAO:  TAO error minimization.
}
The deltaE functions calculate the influence functions of their error criteria.
}

\usage{
error.LMS(arguments)
error.LMLS(arguments)
error.TAO(arguments)
deltaE.LMS(arguments)
deltaE.LMLS(arguments)
deltaE.TAO(arguments)
}
\arguments{
\item{arguments}{List of arguments to pass to the functions.
\itemize{   
\item The first element is the prediction of the neuron.
   \item The second element is the corresponding component of the target vector.
   \item The third element is the whole net. This allows the TAO criterium to know the value of the S parameter and eventually ( next minor update) will allow the user to apply regularization criteria.}
}
}
\value{This functions return the error and influence function criteria.}
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

