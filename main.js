
function main() {
  var dataPath = "plot_data/anatomy/flashcard_difficulty_and_commonness_correlation_anatomy_answers-0.csv";
  var fieldX = 'difficulty';
  var fieldY = 'commonness';
  var plotId = 'plot_comm';
  makeScatterPlot(dataPath, fieldX, fieldY, plotId);

  dataPath = "plot_data/anatomy/flashcard_difficulty_and_learning_rate_correlation_anatomy_answers-0.csv";
  fieldY = 'learning rate';
  plotId = 'plot_learn';
  makeScatterPlot(dataPath, fieldX, fieldY, plotId);
}

function makeScatterPlot(dataPath, fieldX, fieldY, plotId) {
  d3.csv(dataPath, function(flashcards) {

    var data = [preprocessData(flashcards, fieldX, fieldY)];
    var dataByXY = {};
    for (var i = 0; i < flashcards.length; i++) {
      fc = flashcards[i];
      dataByXY[fc[fieldX] + ',' + fc[fieldY]] = fc;
      fc.prediction = 1 / (1 + Math.exp(fc.difficulty));
    }

      Plotly.newPlot(plotId, data, {
        height: window.innerHeight * 0.8,
        xaxis: {
          title: fieldX,
          showgrid: false,
          zeroline: false
        },
        yaxis: {
          title: fieldY,
          showline: false,
          zeroline: false
        }
      });
      var myPlot = document.getElementById(plotId);
      myPlot.on('plotly_click', function(data){
          var pts = '';
          for(var i=0; i < data.points.length; i++){
            var flashcard = dataByXY[data.points[i].x + ',' + data.points[i].y];
            var url = "http://practiceanatomy.com/view/?termsLang=en&context=" + flashcard.context_identifier;
            console.log(flashcard);
            //window.open(url,'_blank');
          }
      });
      showContexts(flashcards);
  });
}

function showContexts(flashcards) {
      var scale = chroma.scale([
          '#e23',
          '#f40',
          '#fa0',
          '#fe3',
          '#5CA03C',
        ]); 


    var contexts = getContexts(flashcards);
    document.getElementById('contexts').innerHTML = contexts.map(function(c) {
      return '<div>' +
        '<h3><a href="http://practiceanatomy.com/view/?termsLang=en&context=' + c.id + '">' +
          c.name + '</a></h3> ' +
        c.flashcards.map(function(fc) {
          return '<span class="label label-default" style="border-bottom: 5px solid ' +
            scale(fc.prediction).hex() + '; margin-bottom: 10px">' +
            fc.term_name + '</span> ';
        }).join('&nbsp; ') +
        '</div>';
    }).join('');
    var div = d3.select("div#contexts");
    var circles = div.selectAll("div")
      .data(contexts)
      .enter()
      .append('a')
      .text(function(d) {
      return d.context_name; });
}

function getContexts(data) {
  var contextsDict = {};
  for (var i = 0; i < data.length; i++) {

    var id = data[i].context_identifier;
    contextsDict[id] = contextsDict[id] || {
      flashcards : [],
      id : id,
      name : data[i].context_name,
    };
    contextsDict[id].flashcards.push(data[i]);
  }
  contexts = [];
  for (i in contextsDict) {
    var diffs = contextsDict[i].flashcards.map(function(fc) {
      return fc.difficulty;
    });
    contextsDict[i].difficultyDiff = diffs.max() - diffs.min();

    contexts.push(contextsDict[i]);
  }
  contexts = contexts.sort(function(a, b) {
    return b.difficultyDiff - a.difficultyDiff;
  });
  return contexts;
}

Array.prototype.max = function() {
  return Math.max.apply(null, this);
};

Array.prototype.min = function() {
  return Math.min.apply(null, this);
};

function preprocessData(data, fieldX, fieldY) {
  return {
    y : data.map(function(row) {
      return row[fieldY];
    }),
    x : data.map(function(row) {
      return row[fieldX];
    }),
    text : data.map(function(row) {
      return row.term_name; // + ' \n(' + row.context_name + ')';
    }),
    textposition: 'middle right',
    mode: 'markers+text',
    type: 'scatter',
  };
}

main();
