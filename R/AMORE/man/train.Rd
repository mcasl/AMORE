\name{train}
\alias{train}

\title{Neural network training function.}

\description{For a given data set (training set), this function modifies the neural network weights and biases to approximate the relationships amongst variables present in the training set. These may serve to satisfy several needs, i.e. fitting non-linear functions.}

\usage{
train(net, P, T, Pval=NULL, Tval=NULL, error.criterium="LMS", report=TRUE,
 n.shows, show.step, Stao=NA,prob=NULL,n.threads=0L)
}

\arguments{
\item{net}{Neural Network to train.}
\item{P}{Training set input values.}
\item{T}{Training set output values}
\item{Pval}{Validation set input values for optional early stopping.}
\item{Tval}{Validation set output values for optional early stopping.}
\item{error.criterium}{Criterium used to measure the goodness of fit:"LMS", "LMLS", "TAO".}
\item{Stao}{Initial value of the S parameter used by the TAO algorithm.}
\item{report}{Logical value indicating whether the training function should keep quiet or should provide graphical/written information during the training process instead.}
\item{n.shows}{Number of times to report (if report is TRUE). The total number of training epochs is n.shows times show.step.}
\item{show.step}{Number of epochs to train non-stop until the training function is allow to report.}
\item{prob}{Vector with the probabilities of each sample so as to apply resampling training.}
\item{n.threads}{Number of threads to spawn for the BATCH* training methods. If <1, spawns NumberProcessors-1 threads. If no OpenMP is found, this argument will be ignored.}
}

\value{This function returns a list with two elements: the trained Neural Network object with weights and biases adjusted by the adaptative backpropagation with momentum method and a matrix with the errors obtained during the training. If the validation set is provided, the early stopping technique is applied.}
\author{
Manuel Castejón Limas.              \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré               \email{j.ordieres@upm.es}\cr
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
\code{\link{newff}} 
}

\keyword{neural}

