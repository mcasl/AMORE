\name{graphviz.MLPnet}
\alias{graphviz.MLPnet}
\title{Neural network graphic representation}
\description{Creates a dot file, suitable to be processed with graphviz, containing a graphical representation of the netwok topology and some numerical information about the network parameters.}
\usage{
graphviz.MLPnet(net, filename, digits)
}

\arguments{
\item{net}{Neural Network.}
\item{filename}{Name of the dot file to be written.}
\item{digits}{Number of digits used to round the parameters.}
}

\value{This function writes a file suitable to be postprocessed with the graphviz package. Thus, multiple formats can be obtained: ps, pdf, ...}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@gmail.com}\cr
Joaquin Ordieres Meré.	            \email{j.ordieres@upm.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unirioja.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{fjmartin@unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}

\references{
http:\/\/www.graphviz.org \cr \cr
}

\keyword{neural}
