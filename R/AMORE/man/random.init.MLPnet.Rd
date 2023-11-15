\name{random.init.MLPnet}
\alias{random.init.MLPnet}
\title{Initialize the network with random weigths and biases.}
\description{Provides random values to the network weights and biases so as to start with. Basically it applies the random.init.MLPneuron function to every neuron in the network.}
\usage{random.init.MLPnet(net) }
\arguments{ \item{net}{The neural network object} }
\value{\emph{random.init.MLPnet} returns the input network with weights and biases changed randomly.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré.	            \email{j.ordieres@upm.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unirioja.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{fjmartin@unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}

\seealso{
\code{\link{random.init.MLPneuron}}, \code{\link{init.MLPneuron}}, \code{\link{newff}}
}
\keyword{neural}
